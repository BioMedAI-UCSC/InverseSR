import os
import torch
import json
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset_v3 import RadioPatchDataset
from models.aekl_no_attention import AutoencoderKL
from utils.config_train import train_config
from utils.config_aekl_v3 import get_hparams
from utils.plot_new import denormalize_data, comparison_plots_ok

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim

def compute_metrics(img_real, img_hat):
    # 1. Conversione a NumPy
    if hasattr(img_real, 'detach'):
        img_real = img_real.detach().cpu().numpy()
    if hasattr(img_hat, 'detach'):
        img_hat = img_hat.detach().cpu().numpy()

    # Assicuriamo la forma 5D: (Batch, Channels, Depth, Height, Width)
    if img_real.ndim == 4:
        img_real = np.expand_dims(img_real, axis=0)
        img_hat = np.expand_dims(img_hat, axis=0)

    batch_size = img_real.shape[0]
    psnr_list = []
    ssim_list = []

    # 2. Iteriamo su OGNI ELEMENTO DEL BATCH
    for b in range(batch_size):
        sample_real = img_real[b] # Shape: (C, D, H, W) oppure (D, H, W)
        sample_rec = img_hat[b]

        # Rimuoviamo eventuali canali/dimensioni singole per il singolo campione
        sample_real = np.squeeze(sample_real).astype(np.float32)
        sample_rec = np.squeeze(sample_rec).astype(np.float32)

        data_range = float(sample_real.max() - sample_real.min())
        
        # Se la patch è interamente vuota/costante, usiamo 1.0 per evitare divisioni per zero
        if data_range == 0 or np.isnan(data_range):
            data_range = 1.0

        # Calcolo PSNR del singolo campione
        psnr_val = skimage_psnr(sample_real, sample_rec, data_range=data_range)
        psnr_list.append(psnr_val)

        # Calcolo SSIM per il singolo campione
        # Se 4D o 3D (C, D, H, W) o (D, H, W), calcoliamo la media fetta per fetta lungo le dimensioni spaziali (H, W)
        if sample_real.ndim >= 3:
            # Le dimensioni spaziali (H, W) sono SEMPRE le ultime due (-2, -1)
            depth_slices = sample_real.reshape(-1, sample_real.shape[-2], sample_real.shape[-1])
            rec_slices = sample_rec.reshape(-1, sample_rec.shape[-2], sample_rec.shape[-1])

            slice_ssims = []
            for d in range(depth_slices.shape[0]):
                s_real = depth_slices[d]
                s_rec = rec_slices[d]

                s_range = float(s_real.max() - s_real.min())
                if s_range == 0:
                    s_range = 1.0

                min_dim = min(s_real.shape)
                if min_dim >= 3:
                    win_size = 7 if min_dim >= 7 else (min_dim if min_dim % 2 != 0 else min_dim - 1)
                    s_val = skimage_ssim(s_real, s_rec, data_range=s_range, win_size=win_size)
                    if not np.isnan(s_val):
                        slice_ssims.append(s_val)

            ssim_val = float(np.mean(slice_ssims)) if len(slice_ssims) > 0 else 0.0
            ssim_list.append(ssim_val)

        elif sample_real.ndim == 2:
            min_dim = min(sample_real.shape)
            win_size = 7 if min_dim >= 7 else (min_dim if min_dim % 2 != 0 else min_dim - 1)
            ssim_val = skimage_ssim(sample_real, sample_rec, data_range=data_range, win_size=win_size)
            ssim_list.append(ssim_val)

    # Restituisce la media di PSNR e SSIM calcolata su tutto il batch
    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))

    return mean_psnr, mean_ssim

def test(hparams, train_param):
    device = train_param.device
    os.makedirs(train_param.test_fig, exist_ok=True)
    print(f"Stai usando il dispositivo: {device}")
    
    # 2. Dataset di Test
    test_dataset = RadioPatchDataset(
            data_dir=train_param.test_dir,
            in_channels=hparams.in_channels,
            norm_mode=train_param.norm_mode,
        )
  
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=train_param.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        )
    
    # 3. Caricamento Modello
    checkpoint_path = os.path.join(train_param.vae_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    hparams_dict = vars(hparams)
    model = AutoencoderKL(embed_dim=hparams.z_channels, hparams=hparams_dict).to(device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    print(f"Pesi caricati con successo da {checkpoint_path}.")
    
    model.eval()

    # 4. Loop di Test
    test_recon_loss = []
    test_psnr_list = []
    test_ssim_list = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader, desc="Testing VAE")):
            x = batch["x_0"].to(device)
            patch_stats = {k: v.to(train_param.device) if isinstance(v, torch.Tensor) else v 
                                                for k, v in batch['stats'].items()
                                                                                        }
            
            # Forward pass
            h = model.encoder(x)
            z = model.quant_conv_mu(h)
            x_hat = model.decode(z)
            
            # MSE Loss sui dati normalizzati
            loss = F.mse_loss(x_hat, x)
            test_recon_loss.append(loss.item())

            

            # 1. Calcola la SSIM sui dati NORMALIZZATI (stabilità numerica perfetta tra 0 e 1)
            _, batch_ssim = compute_metrics(x, x_hat)


            # Denormalizzazione per metriche visive e fisiche
            x_denorm = denormalize_data(x, train_param.norm_mode, patch_stats=patch_stats)
            x_hat_denorm = denormalize_data(x_hat, train_param.norm_mode, patch_stats=patch_stats)
            batch_psnr, _ = compute_metrics(x_denorm, x_hat_denorm)
            # Calcolo PSNR e SSIM sui dati denormalizzati
            batch_psnr, _ = compute_metrics(x_denorm, x_hat_denorm)
            test_psnr_list.append(batch_psnr)
            test_ssim_list.append(batch_ssim)

            # Salvataggio plot ogni 10 batch o all'ultimo
            if i % 10 == 0 or i == len(test_dataloader) - 1:
                fig = comparison_plots_ok(
                    x_denorm, 
                    x_hat_denorm
                )
                fig.savefig(f"{train_param.test_fig}/test_vae_recon_batch_{i}_{train_param.norm_mode}.png")
                plt.close(fig)  # Liberiamo la memoria della figura
            
            # Fondamentale per non saturare la RAM
            del x, x_hat, h, z, x_denorm, x_hat_denorm

    print(f"\n=================== Risultati Test VAE ===================")
    print(f" Average MSE (normalized) : {np.mean(test_recon_loss):.6f}")
    print(f" Average PSNR (denorm)    : {np.mean(test_psnr_list):.2f} dB")
    print(f" Average SSIM (norm)    : {np.mean(test_ssim_list):.4f}")
    print(f"==========================================================\n")


if __name__ == "__main__":
    hparams, _ = get_hparams()
    train_param, _ = train_config()
    test(hparams, train_param)