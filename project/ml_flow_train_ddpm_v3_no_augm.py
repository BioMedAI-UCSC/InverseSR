import os
import json
import copy  # Importato per clonare lo state_dict del miglior modello
from datetime import datetime
from torch.amp import autocast
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path

from models.aekl_no_attention import AutoencoderKL
from models.ddim import DDIMSampler
from models.ddpm_v2_conditioned import DDPM
from utils.config_aekl_v3 import get_hparams
from utils.config_train import train_config
from utils.config_unet_v3 import get_config
from utils.dataset_v3 import RadioPatchDataset
from utils.plot_new import comparison_plots_ok, denormalize_data


# --- CONFIGURAZIONE PERCORSI E DIRECTORY ---
train_cfg, _ = train_config()
hparams, _ = get_hparams()

BASE_SCRATCH = "/leonardo_scratch/large/userexternal/gvitanza/InverseSR/"
RUN_DIR = train_cfg.output_dir_ddpm
current_time = datetime.now().strftime("%b%d_%H-%M-%S")
train_stats_path = os.path.join(train_cfg.data_dir, "train/npy_patches/dataset_filter_stats.json")
val_stats_path = os.path.join(train_cfg.data_dir, "val/npy_patches/dataset_filter_stats.json")

if os.path.exists(train_stats_path):
    with open(train_stats_path, "r") as f:
        train_stats = json.load(f)
else:
    raise FileNotFoundError(f"File {train_stats_path} non trovato! Esegui prima precompute_stats.py")

if os.path.exists(val_stats_path):
    with open(val_stats_path, "r") as f:
        val_stats = json.load(f)
else:
    raise FileNotFoundError(f"File {val_stats_path} non trovato! Esegui prima precompute_stats.py")

print("Inizio configurazione MLFlow...")
db_path = "mlruns_ddpm.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment(
    f"Radio_DDPM_v2_{train_cfg.epochs}epochs_z{hparams.z_channels}_{current_time}"
)

print(f"Caricamento VAE pre-addestrato da {train_cfg.vae_path}")
vae = AutoencoderKL(embed_dim=hparams.z_channels, hparams=vars(hparams))

# OPTIMIZATION OOM: Caricamento prima su CPU per non occupare VRAM
checkpoint_vae = torch.load(
    train_cfg.vae_path, map_location="cpu", weights_only=False
)
vae.load_state_dict(checkpoint_vae["model_state_dict"])

# OPTIMIZATION OOM: Spostamento su GPU, congelamento gradienti e eval mode
vae = vae.to(train_cfg.device)
vae.eval()
vae.requires_grad_(False)
del checkpoint_vae
torch.cuda.empty_cache()

def save_loss_curves(
    history_train_loss, 
    history_val_loss, 
    checkpoint_dir, 
    writer=None, 
    mlflow_instance=None
):
    """
    Genera e salva il grafico Train vs Validation Loss.
    Funziona con qualsiasi numero di epoche completate finora.
    """
    if len(history_train_loss) == 0:
        print("[Warning] Nessuna metrica di loss disponibile per il plot.")
        return

    print("Generazione grafico Train vs Validation Loss...")
    fig_loss, ax = plt.subplots(figsize=(10, 6))
    
    # Usa la lunghezza reale delle metriche raccolte fino a questo momento
    current_epochs = len(history_train_loss)
    epochs_range = range(1, current_epochs + 1)
    
    ax.plot(epochs_range, history_train_loss, label='Training Loss', color='tab:blue', linewidth=2)
    
    # Se ci sono metriche di validazione
    if len(history_val_loss) > 0:
        val_epochs_range = range(1, len(history_val_loss) + 1)
        ax.plot(val_epochs_range, history_val_loss, label='Validation Loss', color='tab:orange', linewidth=2)
        
        # Individua l'epoca migliore basandosi sulla val_loss reale raccolta
        best_idx = int(np.argmin(history_val_loss))
        best_epoch = best_idx + 1
        best_val_loss = history_val_loss[best_idx]
        
        # Evidenzia l'epoca ottimale
        ax.scatter(best_epoch, best_val_loss, color='red', s=80, zorder=5, label=f'Best Epoch: {best_epoch}')
        ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)

    ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoca', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=11)
    
    fig_loss.tight_layout()
    
    # Salvataggio su disco
    os.makedirs(checkpoint_dir, exist_ok=True)
    loss_plot_path = os.path.join(checkpoint_dir, "loss_curves.png")
    fig_loss.savefig(loss_plot_path, dpi=300)
    
    # Log su TensorBoard / MLflow se disponibili
    if writer is not None:
        writer.add_figure("Visual/Loss_Curves", fig_loss, global_step=current_epochs)
        
    if mlflow_instance is not None:
        try:
            mlflow_instance.log_artifact(loss_plot_path, artifact_path="plots")
        except Exception as e:
            print(f"Warning: Impossibile inviare l'artifact a MLflow: {e}")

    plt.close(fig_loss)
    print(f"✅ Grafico salvato correttamente in: {loss_plot_path}")

def train():
    dataset_name = Path(train_cfg.data_dir).name
    CHECKPOINT_DIR = os.path.join(
        BASE_SCRATCH,
        f"ddpm_{train_cfg.cond_key}_{dataset_name}_{train_cfg.epochs}epochs_{train_cfg.norm_mode}_{current_time}",
    )
    TB_LOG_DIR = train_cfg.tensor_board_logger_ddpm
    log_dir = f"{TB_LOG_DIR}/run_{current_time}_lr_{train_cfg.learning_rate}_z{hparams.z_channels}"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 1. Configurazione UNet
    unet_cfg, _ = get_config()

    use_mask_channel = unet_cfg["params"]["use_mask_channel"]
    in_ch = hparams.z_channels + (1 if use_mask_channel else 0)

    unet_cfg["params"]["in_channels"] = in_ch
    unet_cfg["params"]["out_channels"] = hparams.z_channels  # Output ricostruisce solo z
    unet_cfg["params"].pop("out_channels_unet", None)
    unet_cfg["params"].pop("in_channels_unet", None)
    
    DATA_DIR = os.path.join(train_cfg.data_dir, "train/npy_patches")
    # 2. Dataset e DataLoader (TRAIN & VALIDATION)
    train_dataset = RadioPatchDataset(
        data_dir=DATA_DIR,
        catalogue_path=train_cfg.catalogue_path,
        in_channels=hparams.in_channels,
        norm_mode=train_cfg.norm_mode,
        cached_stats=train_stats["flux_statistics_filtered"],
    )
    


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    val_dataset = RadioPatchDataset(
        data_dir=os.path.join(train_cfg.data_dir, "val/npy_patches"),
        in_channels=hparams.in_channels,
        norm_mode=train_cfg.norm_mode,
        cached_stats=val_stats["flux_statistics_filtered"],
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )


    # 3. Modello DDPM Latente, Ottimizzatore e Scheduler
    model = DDPM(
        unet_config=unet_cfg,
        conditioning_key=train_cfg.cond_key,
        learn_logvar=True,
    ).to(train_cfg.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.learning_rate
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.epochs,
        eta_min=1e-6
    )

    writer = SummaryWriter(log_dir=log_dir)

    # Liste per tracciare lo storico delle loss per ogni epoca
    history_train_loss = []
    history_val_loss = []

    # Variabili per il tracciamento del miglior modello
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_weights = None  # Salva in RAM lo state_dict del miglior modello
    try:
        with mlflow.start_run(run_name=f"DDPM_Training_{current_time}"):
            mlflow.log_params(vars(train_cfg))
            mlflow.log_params({f"vae_{k}": v for k, v in vars(hparams).items()})
            mlflow.log_params(
                {f"unet_{k}": v for k, v in unet_cfg["params"].items()}
            )

            for epoch in range(train_cfg.epochs):
                # ==================== FASE DI TRAINING ====================
                model.train()
                pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} [Train]")
                epoch_loss = []

                for batch in pbar:
                    optimizer.zero_grad()

                for step, batch in enumerate(pbar):
                    x_start = batch["x_0"].to(train_cfg.device)
                    spatial_mask = batch["spatial_mask"].to(train_cfg.device)
                    context = batch["context"].to(train_cfg.device)

                    # STEP 1: Encoding Latente via VAE
                    with torch.no_grad():
                        h = vae.encoder(x_start)
                        z = vae.quant_conv_mu(h)

                    # STEP 2: Preparazione Condizionamento
                    cond_payload = {}

                    if train_cfg.cond_key in ["concat", "hybrid"]:
                        latent_shape = z.shape[2:]
                        mask_latent = F.interpolate(
                            spatial_mask,
                            size=latent_shape,
                            mode="trilinear",
                            align_corners=False,
                        )
                        cond_payload["c_concat"] = [mask_latent]

                    if (
                        train_cfg.cond_key in ["crossattn", "hybrid"]
                        and context is not None
                    ):
                        context_attn = (
                            context.unsqueeze(1)
                            if context.ndim == 2
                            else context
                        )
                        cond_payload["c_crossattn"] = [context_attn]

                    # STEP 3: Loss e Backward
                    loss, _ = model(z, cond_payload)
                    loss.backward()
                    optimizer.step()

                    epoch_loss.append(loss.item())
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()

                avg_train_loss = np.mean(epoch_loss)
                history_train_loss.append(avg_train_loss)

                writer.add_scalar("Loss/Train_DDPM", avg_train_loss, epoch)
                writer.add_scalar("Params/LearningRate", current_lr, epoch)
                mlflow.log_metric("avg_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("learning_rate", current_lr, step=epoch)

                # ==================== FASE DI VALIDAZIONE ====================
                model.eval()
                val_losses = []

                with torch.no_grad():
                    for val_batch in val_dataloader:
                        x_val = val_batch["x_0"].to(train_cfg.device)
                        mask_val = val_batch["spatial_mask"].to(train_cfg.device)
                        context_val = val_batch["context"].to(train_cfg.device)

                        h_val = vae.encoder(x_val)
                        z_val = vae.quant_conv_mu(h_val)

                        val_cond_payload = {}
                        if train_cfg.cond_key in ["concat", "hybrid"]:
                            mask_latent_val = F.interpolate(
                                mask_val,
                                size=z_val.shape[2:],
                                mode="trilinear",
                                align_corners=False,
                            )
                            val_cond_payload["c_concat"] = [mask_latent_val]

                        if (
                            train_cfg.cond_key in ["crossattn", "hybrid"]
                            and context_val is not None
                        ):
                            context_attn_val = (
                                context_val.unsqueeze(1)
                                if context_val.ndim == 2
                                else context_val
                            )
                            val_cond_payload["c_crossattn"] = [context_attn_val]

                        v_loss, _ = model(z_val, val_cond_payload)
                        val_losses.append(v_loss.item())

                avg_val_loss = np.mean(val_losses)
                history_val_loss.append(avg_val_loss)

                writer.add_scalar("Loss/Val_DDPM", avg_val_loss, epoch)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

                # --- SALVATAGGIO BEST MODEL ---
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    best_ckpt_path = os.path.join(CHECKPOINT_DIR, "ddpm_best_epoch.pth")

                    # Mantiene in memoria lo stato dei pesi migliori per MLFlow
                    best_model_weights = copy.deepcopy(model.state_dict())

                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "best_val_loss": best_val_loss,
                        },
                        best_ckpt_path,
                    )
                    print(f" -> [Best Model] Salvato nuovo miglior checkpoint all'epoca {best_epoch} con Val Loss: {best_val_loss:.6f}")

                # ==================== GENERAZIONE E PLOT ====================
                if (epoch % 1 == 0 and epoch <= 20) or (epoch % 20 == 0 and epoch > 20):
                    with torch.no_grad():
                        val_sample_batch = next(iter(val_dataloader))
                        x_vis = val_sample_batch["x_0"][:2].to(train_cfg.device)
                        patch_stats = {
                            k: v[:2].to(train_cfg.device) if isinstance(v, torch.Tensor) else v
                            for k, v in val_sample_batch['stats'].items()
                        }
                        mask_vis = val_sample_batch["spatial_mask"][:2].to(train_cfg.device)
                        context_vis = val_sample_batch["context"][:2].to(train_cfg.device)

                        eval_batch_size = x_vis.shape[0]

                        h_vis = vae.encoder(x_vis)
                        z_vis = vae.quant_conv_mu(h_vis)
                        _, _, D_lat, H_lat, W_lat = z_vis.shape
                        shape = (hparams.z_channels, D_lat, H_lat, W_lat)

                        vis_cond_payload = {}
                        if train_cfg.cond_key in ["concat", "hybrid"]:
                            mask_latent_vis = F.interpolate(
                                mask_vis,
                                size=(D_lat, H_lat, W_lat),
                                mode="trilinear",
                                align_corners=False,
                            )
                            vis_cond_payload["c_concat"] = [mask_latent_vis]

                        if (
                            train_cfg.cond_key in ["crossattn", "hybrid"]
                            and context_vis is not None
                        ):
                            context_attn_vis = (
                                context_vis.unsqueeze(1)
                                if context_vis.ndim == 2
                                else context_vis
                            )
                            vis_cond_payload["c_crossattn"] = [context_attn_vis]

                        sampler = DDIMSampler(model)
                        img_noise = torch.randn(
                            (eval_batch_size, *shape), device=train_cfg.device
                        )

                        z_gen, _ = sampler.sample(
                            S=50,
                            batch_size=eval_batch_size,
                            shape=shape,
                            conditioning=vis_cond_payload,
                            first_img=img_noise,
                            eta=0.0,
                            verbose=False,
                        )

                        x_gen = vae.decode(z_gen)

                        x_real_denorm = denormalize_data(x_vis, train_cfg.norm_mode, patch_stats=patch_stats)
                        x_gen_denorm = denormalize_data(x_gen, train_cfg.norm_mode, patch_stats=patch_stats)

                        mask_sample = mask_vis[0, 0].cpu().numpy()
                        src_z, src_y, src_x = np.where(mask_sample > 0.5)
                        coords = list(zip(src_x, src_y, src_z))

                        try:
                            fig = comparison_plots_ok(
                                x_real_denorm[0],
                                x_gen_denorm[0],
                                sources_coords=coords
                            )
                            writer.add_figure(
                                "Visual/3D_Comparison", fig, global_step=epoch
                            )
                            plt.close(fig)
                        except Exception as e:
                            print(
                                f"[Warning] Errore durante la generazione del plot all'epoca {epoch}: {e}"
                            )
                            plt.close("all")
                    model.train()

                # --- ULTIMO CHECKPOINT ---
                if (epoch + 1) == train_cfg.epochs:
                    ckpt_path = os.path.join(
                        CHECKPOINT_DIR, f"ddpm_ep{epoch+1}.pth"
                    )
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "val_loss": avg_val_loss,
                        },
                        ckpt_path,
                    )

            # --- SALVATAGGIO FINALE & METRICHE SU MLFLOW ---
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_param("best_epoch", best_epoch)

            # 1. Registrazione del BEST MODEL su MLFlow
            if best_model_weights is not None:
                print(f"Registrazione su MLFlow del BEST MODEL dall'epoca {best_epoch}...")
                model.load_state_dict(best_model_weights)

                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="ddpm_best",
                    registered_model_name=f"DDPM_Best_{hparams.z_channels}ch",
                )
                local_best_model_path = os.path.join(RUN_DIR, "ddpm_best_model")
                mlflow.pytorch.save_model(model, path=local_best_model_path)
                if os.path.exists(best_ckpt_path):
                    mlflow.log_artifact(best_ckpt_path, artifact_path="checkpoints")

                # 2. Registrazione del MODELLO FINALE (ultima epoca) su MLFlow
                print("Registrazione modello DDPM finale (ultima epoca)...")
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="ddpm_final",
                    registered_model_name=f"DDPM_Final_{hparams.z_channels}ch",
                )

                local_model_path = os.path.join(RUN_DIR, "ddpm_final_model")
                mlflow.pytorch.save_model(model, path=local_model_path)


                writer.close()
                print(f"Training concluso. Checkpoint in {CHECKPOINT_DIR}")
                print(f"Miglior modello salvato all'epoca {best_epoch} con validation loss pari a {best_val_loss:.6f}")
    except KeyboardInterrupt:
        print("\nTraining interrotto dall'utente.")
    except Exception as e:
        print(f"\nTraining interrotto da un errore: {e}")
    finally:
        print("Salvataggio finale delle curve di loss...")
        save_loss_curves(history_train_loss, history_val_loss, CHECKPOINT_DIR, writer, mlflow)
            

if __name__ == "__main__":
    train()