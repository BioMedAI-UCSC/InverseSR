import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import mlflow 
import json

from torch.utils.data import DataLoader
from utils.dataset_v3 import RadioPatchDataset
from utils.config_train import train_config
from utils.config_unet_v3 import get_config
from utils.config_aekl_v3 import get_hparams
from models.aekl_no_attention import AutoencoderKL
from models.ddpm_v2_conditioned import DDPM

from models.ddim import DDIMSampler
from utils.plot_new import denormalize_data, comparison_plots_ok





def quick_test_metrics(model, vae, dataloader, train_param, hparams, unet_cfg):
    model.eval()
    vae.eval()
     
    os.makedirs(train_param.test_fig, exist_ok=True)
    sampler = DDIMSampler(model)
    device = train_param.device
    epoch = train_param.epochs
    
    mses, ssims, psnrs = [], [], []
    
    # Parametri di shape fissi
    latent_size = 128 // 4
    shape = (hparams.z_channels, latent_size, latent_size, latent_size)
    ddim_steps = 50 
    use_mask_channel = unet_cfg["params"]["use_mask_channel"]  # Assicurati sia booleano dal tuo argparser
    
    with torch.no_grad():
        print(f"Totale campioni nel dataset: {len(dataloader)}")
        print(f"Batch size: {train_param.batch_size}")
        print(f"Totale batch (lunghezza dataloader): {len(dataloader)}")
        for batch_idx, batch in enumerate(dataloader):
            x_start = batch["x_0"].to(device)
            patch_stats = {k: v.to(train_param.device) if isinstance(v, torch.Tensor) else v 
                                                            for k, v in batch['stats'].items()
                                                                                                    }
            spatial_mask = batch.get("spatial_mask", None)
            if spatial_mask is not None:
                spatial_mask = spatial_mask.to(device)
                
            context = batch.get("context", None)
            if context is not None:
                context = context.to(device)

            # --- 1. ENCODING LATENTE VIA VAE ---
            h = vae.encoder(x_start)
            z = vae.quant_conv_mu(h)
            

            # --- 2. PREPARAZIONE MASK LATENT (Se abilitata) ---
            mask_latent = None
            if use_mask_channel and spatial_mask is not None:
                latent_shape = z.shape[2:]  # (D_lat, H_lat, W_lat)
                mask_latent = F.interpolate(
                    spatial_mask, 
                    size=latent_shape, 
                    mode='trilinear', 
                    align_corners=False
                )

            # --- 3. COSTRUZIONE COND_PAYLOAD ---
            cond_payload = {}
            
            # Mode A: Concat (es. Maschera)
            if train_param.cond_key in ["concat", "hybrid"]:
                if spatial_mask is not None:
                    latent_shape = z.shape[2:]  # (D_lat, H_lat, W_lat)
                    mask_latent = F.interpolate(spatial_mask, size=latent_shape, mode='trilinear', align_corners=False)
                    cond_payload["c_concat"] = [mask_latent]

            # Mode B: Cross-Attention (es. Metadati / Context)
            if train_param.cond_key in ["crossattn", "hybrid"]:
                if context is not None:
                    context_attn = context.unsqueeze(1) if context.ndim == 2 else context
                    cond_payload["c_crossattn"] = [context_attn]

            # Gestione Unconditional o fallback
            if train_param.cond_key in [None, "None", "none"] or not cond_payload:
                conditioning = None
            else:
                conditioning = cond_payload
            

            # --- 4. SAMPLING CON DDIM ---
            current_batch_size = x_start.shape[0]
            _, _, D_lat, H_lat, W_lat = z.shape
            shape = (hparams.z_channels, D_lat, H_lat, W_lat)
            
            sampler = DDIMSampler(model)
            img_noise = torch.randn((current_batch_size, *shape), device=train_param.device)

            # Passa cond_payload (oppure None se cond_key == "None")
            conditioning = cond_payload if train_param.cond_key not in [None, "None", "none"] else None

            z_gen, _ = sampler.sample(
                S=ddim_steps,
                batch_size=current_batch_size,
                shape=shape,
                conditioning=conditioning,  # <-- Usa il payload corretto!
                first_img=img_noise,
                eta=0.0,
                verbose=False
            )

            # --- 5. DECODING VAE E METRICHE ---
            z_gen_unscaled = z_gen 
            x_gen = vae.decode(z_gen_unscaled) # x_gen è normalizzato [0, 1]

            # Denormalizzazione corretta per calcolare MSE/PSNR in unità fisiche
            x_start_denorm = denormalize_data(x_start, train_param.norm_mode, patch_stats=patch_stats)
            x_gen_denorm = denormalize_data(x_gen, train_param.norm_mode, patch_stats=patch_stats)
            
            # --- SALVATAGGIO PLOTS COMPARATIVI ---
            if batch_idx % 30 == 0 or batch_idx == len(dataloader) - 1:
                name_real = batch.get("name_x0", [f"Patch_x0_{batch_idx}"])[0]
                
                if train_param.cond_key in [None, "None", "none"]:
                    title_r = f"Real Target: {name_real}"
                    title_g = f"Unconditioned Sample (Epoch {train_param.epochs})"
                else:
                    name_cond = batch.get("name_context", [f"Patch_context_{batch_idx}"])[0]
                    title_r = f"Real Target: {name_real}"
                    title_g = f"Generated from: {name_cond}"
                
                # Genera il grafico usando le immagini denormalizzate
                fig = comparison_plots_ok(
                    x_start_denorm, 
                    x_gen_denorm, 
                )
                
                save_path = os.path.join(train_param.test_fig, f"{train_param.norm_mode}_batch_{batch_idx}.png")
                fig.savefig(save_path)
                plt.close(fig)
                print(f"Salvata figura di confronto in: {save_path}")
            # Ciclo sui campioni del batch per le metriche
            for i in range(current_batch_size):
                # 1. Dati normalizzati per SSIM (assicuriamo conversioni sicure a numpy)
                real_norm = x_start[i, 0].cpu().numpy() if hasattr(x_start, 'cpu') else x_start[i, 0]
                gen_norm = x_gen[i, 0].cpu().numpy() if hasattr(x_gen, 'cpu') else x_gen[i, 0]
                gen_norm = np.clip(gen_norm, 0, 1)

                # 2. Dati denormalizzati per PSNR / MSE
                real_phys = x_start_denorm[i, 0].cpu().numpy() if hasattr(x_start_denorm, 'cpu') else x_start_denorm[i, 0]
                gen_phys = x_gen_denorm[i, 0].cpu().numpy() if hasattr(x_gen_denorm, 'cpu') else x_gen_denorm[i, 0]

                # 3. Estrazione slice centrale 2D per il calcolo
                mid = real_norm.shape[0] // 2
                
                # MSE e PSNR su dati fisici
                mses.append(mse(real_phys[mid], gen_phys[mid]))
                
                p_range = float(real_phys[mid].max() - real_phys[mid].min())
                if p_range == 0 or np.isnan(p_range): 
                    p_range = 1.0
                psnrs.append(psnr(real_phys[mid], gen_phys[mid], data_range=p_range))

                # SSIM sui dati normalizzati [0, 1]
                ssims.append(ssim(real_norm[mid], gen_norm[mid], data_range=1.0))

        # 5. Aggregazione finale di tutti gli elementi di tutti i batch
        metrics = {
            "test/mse": np.mean(mses),
            "test/ssim": np.mean(ssims),
            "test/psnr": np.mean(psnrs)
        }
        
        print(f"\n--- GLOBAL TEST METRICS EPOCH {epoch} ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}")
            try:
                mlflow.log_metric(k, v, step=epoch)
            except:
                pass
        
        return metrics

if __name__ == "__main__":
    # Esempio di utilizzo
  

    # Setup dataset e dataloader
    train_param, _ = train_config()
    hparams, _ = get_hparams()
    unet_cfg, _ = get_config()
    unet_cfg["params"]["in_channels"] = unet_cfg["params"]["in_channels_unet"]
    unet_cfg["params"]["out_channels"] = unet_cfg["params"]["out_channels_unet"]
    unet_cfg["params"].pop("out_channels_unet", None)  # Rimuoviamo i parametri specifici del config per evitare confusione
    unet_cfg["params"].pop("in_channels_unet", None)
    test_stats_path = os.path.join(train_param.test_dir, "dataset_filter_stats.json")
    if os.path.exists(test_stats_path):
        with open(test_stats_path, "r") as f:
            test_stats = json.load(f)
    else:
        raise FileNotFoundError(f"File {test_stats_path} non trovato! Esegui prima precompute_stats.py")
    test_dataset = RadioPatchDataset(
                data_dir=train_param.test_dir,
                in_channels=hparams.in_channels,
                norm_mode=train_param.norm_mode,
                cached_stats=test_stats["flux_statistics_filtered"],
            )
      
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=train_param.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        )

    # Inizializza VAE e DDPM 
    vae = AutoencoderKL(embed_dim=hparams.z_channels, hparams=vars(hparams)).to(train_param.device)
    vae_path = train_param.vae_path
    
    if os.path.exists(vae_path):
        vae_checkpoint = torch.load(vae_path, map_location=train_param.device, weights_only=False)
        
        if isinstance(vae_checkpoint, dict) and "model_state_dict" in vae_checkpoint:
            state_dict = vae_checkpoint["model_state_dict"]
            print("Estratto 'model_state_dict' dal checkpoint globale.")
        else:
            state_dict = vae_checkpoint
        vae.load_state_dict(state_dict)
        print(f"Pesi del VAE caricati con successo da {vae_path}!")
    else:
        print(f"ATTENZIONE: Checkpoint VAE non trovato in {vae_path}!")
    

    model = DDPM(
        unet_config=unet_cfg,
        conditioning_key=train_param.cond_key, 
        learn_logvar=True
    ).to(train_param.device)
    ddpm_path = train_param.output_dir_ddpm
    if os.path.exists(ddpm_path):
        ddpm_checkpoint = torch.load(ddpm_path, map_location=train_param.device, weights_only=False)
        # Controlla se i pesi sono dentro 'model_state_dict' (come dice l'errore)
        if isinstance(ddpm_checkpoint, dict) and "model_state_dict" in ddpm_checkpoint:
            state_dict = ddpm_checkpoint["model_state_dict"]
            print("Estratto 'model_state_dict' dal checkpoint globale.")
        else:
            state_dict =ddpm_checkpoint
        # Adatta in base a come salvi il dizionario di stato (es. checkpoint['model_state_dict'])
        model.load_state_dict(state_dict)
        print(f"Pesi del DDPM caricati con successo da {ddpm_path}!")
    else:
        print(f"ATTENZIONE: Checkpoint DDPM non trovato in {ddpm_path}!")
    # Esegui il test rapido
    quick_test_metrics(model, vae, test_dataloader, train_param, hparams=hparams, unet_cfg=unet_cfg)