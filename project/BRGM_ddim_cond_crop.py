# Code adapted for Astrophysical Data Restoration
# Original Reference: Pinaya et al. (2022) & Marinescu et al. (2020)

import csv
import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

from datetime import datetime
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models.BRGM.forward_models import ForwardDownsample
from models.ddim import DDIMSampler
from utils.add_argument import add_argument
from utils.const import INPUT_FOLDER_CAT
from utils.dataset_stats import compute_dataset_stats
from utils.dataset_v3 import normalize_dynamic
from utils.early_stop import LatentEarlyStopping
from utils.metrics_benchmark import run_astrophysical_benchmarks
from utils.plot_new import (
    compare_cubes,
    comparison_plots_ok,
    denormalize_data,
    draw_img,
    plot_orthogonal_cuts,
)
from utils.power_spectrum_new import (
    compute_power_spectrum_and_cross,
    save_power_spectrum_plots,
)
from utils.utils_new import (
    create_corruption_function,
    getVggFeatures,
    load_pre_trained_model,
    load_vgg_perceptual,
    sampling_from_ddim,
    setup_noise_inputs,
)


# =============================================================================
# HELPER FUNCTIONS & UTILITIES
# =============================================================================
def logprint(message: str, verbose: bool) -> None:
    if verbose:
        print(message)


def denormalize_cond(cond: torch.Tensor, catalogue: pd.DataFrame, feature_cols: list) -> torch.Tensor:
    """Denormalizes parameters based on physical values from the catalogue."""
    mins = [catalogue[col].min() for col in feature_cols]
    maxs = [catalogue[col].max() for col in feature_cols]
    
    stats_min = torch.tensor(mins, device=cond.device, dtype=torch.float32)
    stats_max = torch.tensor(maxs, device=cond.device, dtype=torch.float32)
    
    current_cond_phys = cond.detach() * (stats_max - stats_min) + stats_min
    return torch.clamp(current_cond_phys, stats_min, stats_max)


def create_mask_for_backprop(hparams: Namespace, device: torch.device) -> torch.Tensor:
    mask_cond = torch.ones((1, 4), device=device)
    mask_cond[:, 0] = 0 if not hparams.update_hi_size else 1
    mask_cond[:, 1] = 0 if not hparams.update_line_flux_integral else 1
    mask_cond[:, 2] = 0 if not hparams.update_i else 1
    mask_cond[:, 3] = 0 if not hparams.update_w20 else 1
    return mask_cond


def crop_center_3d(tensor_or_array, crop_size):
    """
    Estrae un ritaglio centrale da un tensor 5D (B, C, D, H, W) 
    o da un array numpy 3D (D, H, W).
    """
    is_numpy = isinstance(tensor_or_array, np.ndarray)
    
    if isinstance(crop_size, (tuple, list)):
        cH, cW = crop_size[0], crop_size[1]
    else:
        cH, cW = crop_size, crop_size

    if is_numpy:
        D, H, W = tensor_or_array.shape
        start_h = max(0, (H - cH) // 2)
        start_w = max(0, (W - cW) // 2)
        return tensor_or_array[:, start_h : start_h + cH, start_w : start_w + cW]
    else:
        _, _, D, H, W = tensor_or_array.shape
        start_h = max(0, (H - cH) // 2)
        start_w = max(0, (W - cW) // 2)
        return tensor_or_array[:, :, :, start_h : start_h + cH, start_w : start_w + cW]


def plot_zoomed_comparison(
    target_phys: np.ndarray,
    synth_phys: np.ndarray,
    save_path: Path,
    crop_size_ratio: float = 0.35,
    filename: str = "sr_zoomed_comparison_paper.png",
) -> None:
    """
    Generates a publication-ready figure comparing HR target and SR output.
    """
    if target_phys.ndim == 3:
        max_idx = np.unravel_index(np.argmax(target_phys), target_phys.shape)
        slice_idx = max_idx[0]
        target_2d = target_phys[slice_idx]
        synth_2d = synth_phys[slice_idx]
    else:
        target_2d = target_phys
        synth_2d = synth_phys

    h, w = target_2d.shape
    max_y, max_x = np.unravel_index(np.argmax(target_2d), (h, w))

    crop_h = int(h * crop_size_ratio)
    crop_w = int(w * crop_size_ratio)

    y1 = max(0, min(h - crop_h, max_y - crop_h // 2))
    y2 = min(h, y1 + crop_h)
    x1 = max(0, min(w - crop_w, max_x - crop_w // 2))
    x2 = min(w, x1 + crop_w)

    target_crop = target_2d[y1:y2, x1:x2]
    synth_crop = synth_2d[y1:y2, x1:x2]

    vmin = min(target_2d.min(), synth_2d.min())
    vmax = max(target_2d.max(), synth_2d.max())

    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.0), dpi=300)

    im0 = axes[0, 0].imshow(target_2d, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Ground Truth (Target HR)", fontweight='bold')
    axes[0, 0].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.8, edgecolor='cyan', facecolor='none', linestyle='--'))
    axes[0, 0].set_xlabel("X [pixels]")
    axes[0, 0].set_ylabel("Y [pixels]")

    axes[0, 1].imshow(synth_2d, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Reconstructed (SR Model)", fontweight='bold')
    axes[0, 1].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.8, edgecolor='cyan', facecolor='none', linestyle='--'))
    axes[0, 1].set_xlabel("X [pixels]")
    axes[0, 1].set_ylabel("Y [pixels]")

    axes[1, 0].imshow(target_crop, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f"Target Zoom (ROI: [{x1}:{x2}, {y1}:{y2}])", fontweight='bold')
    axes[1, 0].set_xlabel("X [pixels]")
    axes[1, 0].set_ylabel("Y [pixels]")

    axes[1, 1].imshow(synth_crop, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("Reconstructed Zoom (ROI)", fontweight='bold')
    axes[1, 1].set_xlabel("X [pixels]")
    axes[1, 1].set_ylabel("Y [pixels]")

    fig.subplots_adjust(right=0.86, hspace=0.28, wspace=0.25)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(im0, cax=cbar_ax)
    cbar.set_label("Physical Intensity / Flux", rotation=270, labelpad=15)

    fig.savefig(save_path / filename, bbox_inches='tight', dpi=300)
    plt.close(fig)


def load_crop_from_path(
    file_path: str,
    top_y: int,
    left_x: int,
    crop_h: int,
    crop_w: int,
    device: torch.device,
    target_depth: int = None,
    norm_mode: str = "global_1",
) -> tuple[torch.Tensor, dict]:
    """
    Carica ed estrae un crop spaziale/spettrale da file .npy.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File non trovato al percorso: {file_path}")

    if file_path.endswith(".npy"):
        data_mmap = np.load(file_path, mmap_mode="r")
        if data_mmap.ndim == 3:
            raw_crop = data_mmap[:, top_y : top_y + crop_h, left_x : left_x + crop_w]
        elif data_mmap.ndim == 4:
            raw_crop = data_mmap[0, :, top_y : top_y + crop_h, left_x : left_x + crop_w]
        elif data_mmap.ndim == 5:
            raw_crop = data_mmap[0, 0, :, top_y : top_y + crop_h, left_x : left_x + crop_w]
        else:
            raise ValueError(f"Formato NPY {data_mmap.ndim}D non supportato.")
        raw_crop = np.nan_to_num(np.array(raw_crop, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        tensor_crop = torch.from_numpy(raw_crop)
    else:
        raise ValueError(f"Estensione file non supportata per: {file_path}")

    if tensor_crop.ndim == 3:
        tensor_crop = tensor_crop.unsqueeze(0).unsqueeze(0)
    elif tensor_crop.ndim == 4:
        tensor_crop = tensor_crop.unsqueeze(0)

    if target_depth is not None and tensor_crop.shape[2] != target_depth:
        tensor_crop = F.interpolate(
            tensor_crop,
            size=(target_depth, crop_h, crop_w),
            mode="trilinear",
            align_corners=False
        )

    crop_stats = compute_dataset_stats(file_path)
    tensor_crop = tensor_crop.to(device)

    if norm_mode != "none":
        clamp_min = -1.0 if norm_mode == "zscore" else 0.0
        tensor_crop, _ = normalize_dynamic(tensor_crop, norm_mode=norm_mode, stats=crop_stats)
        tensor_crop = torch.clamp(tensor_crop, clamp_min, 1.0)

    return tensor_crop, crop_stats


# =============================================================================
# 1. MOTORE LIGHTWEIGHT DI OTTIMIZZAZIONE LATENTE
# =============================================================================
def optimize_latent_patch(
    ddim: DDIMSampler,
    decoder: torch.nn.Module,
    forward: ForwardDownsample,
    target_img_corrupted: torch.Tensor,
    device: torch.device,
    hparams: Namespace,
    patch_size: tuple = (128, 128, 128),
    cat: dict = None,
    mask_latent: torch.Tensor = None,
    vgg16: torch.nn.Module = None,
    target_features: torch.Tensor = None,
    writer: SummaryWriter = None,
    patch_tag: str = "Patch_0",
    batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    
    update_params = []
    cond = None

    if hparams.cond_key in ["concat", "hybrid", "crossattn"]:
        cond, latent_variable = setup_noise_inputs(cat, device=device, hparams=hparams, batch_size=batch_size, target_patch_shape=patch_size)
        if getattr(hparams, "update_conditioning", False):
            cond = cond.detach().requires_grad_(True)
            update_params.append(cond)
    else:
        _, latent_variable = setup_noise_inputs(cat, device=device, hparams=hparams, batch_size=batch_size, target_patch_shape=patch_size)

    if getattr(hparams, "update_latent_variables", True):
        latent_variable = latent_variable.detach().requires_grad_(True)
        update_params.append(latent_variable)

    optimizer, scheduler = None, None
    if len(update_params) > 0:
        optimizer = torch.optim.Adam(update_params, betas=(0.9, 0.999), lr=hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=hparams.num_steps - hparams.start_steps, eta_min=1e-6
        )

    early_stopper = LatentEarlyStopping(patience=hparams.patience, min_delta=hparams.min_delta, save_to_disk=False)
    
    # =========================================================================
    # INIZIALIZZAZIONI E PRE-CALCOLI FUORI DAL CICLO FOR
    # =========================================================================
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    # Pre-calcolo statico della matrice dei pesi della pixel loss
    weights = torch.abs(target_img_corrupted) + 1.0

    # Pre-strutturazione dinamica delle condizioni per il payload
    has_concat = hparams.cond_key in ["concat", "hybrid"]
    has_crossattn = hparams.cond_key in ["crossattn", "hybrid"]
    update_cond_grad = getattr(hparams, "update_conditioning", False)
    
    if update_cond_grad and cond is not None:
        mask_backprop = create_mask_for_backprop(hparams, device)
    # =========================================================================

    for step in range(hparams.start_steps, hparams.num_steps):
        if optimizer is not None:
            optimizer.zero_grad()

        # 1. Forward Pass & Loss Calculation in Mixed Precision
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            cond_payload = {}
            if has_concat:
                cond_payload["c_concat"] = [mask_latent]
            if has_crossattn:
                context_attn = cond.unsqueeze(1) if (cond is not None and cond.ndim == 2) else cond
                cond_payload["c_crossattn"] = [context_attn]

            synth_img = sampling_from_ddim(
                ddim=ddim, 
                decoder=decoder, 
                latent_variable=latent_variable, 
                cond=cond_payload, 
                hparams=hparams
            )
            synth_img_corrupted = forward(synth_img)

            if synth_img_corrupted.shape != target_img_corrupted.shape:
                synth_img_corrupted = F.interpolate(
                    synth_img_corrupted,
                    size=target_img_corrupted.shape[2:],
                    mode="trilinear",
                    align_corners=False
                )

            # Uso dei pesi pre-calcolati esternamente al loop
            pixel_loss = ((synth_img_corrupted - target_img_corrupted).abs() * weights).mean()
            loss = pixel_loss

            if hasattr(hparams, "lambda_prior") and hparams.lambda_prior > 0:
                loss += hparams.lambda_prior * torch.mean(latent_variable ** 2)

            if hparams.lambda_perc > 0 and vgg16 is not None and target_features is not None:
                synth_features = getVggFeatures(hparams, synth_img_corrupted, vgg16)
                perc_loss = (target_features - synth_features).abs().mean()
                loss += hparams.lambda_perc * perc_loss

        current_loss = loss.item()

        # 2. Backward Pass & Optimizer Step scalato
        if len(update_params) > 0 and loss.requires_grad:
            if step % hparams.prior_every == 0:
                scaler.scale(loss).backward()

                if update_cond_grad and cond is not None and cond.grad is not None:
                    scaler.unscale_(optimizer)
                    cond.grad *= mask_backprop

                scaler.step(optimizer)
                scaler.update()

        # Logging della Loss su TensorBoard
        if writer is not None:
            writer.add_scalar(f"Optimization_Loss/{patch_tag}", current_loss, global_step=step)

        if scheduler is not None:
            scheduler.step()

        if hparams.cond_key != "None" and cond is not None:
            with torch.no_grad():
                cond.clamp_(0, 1)

        early_stopper(current_loss, latent_variable, cond if hparams.cond_key != "None" else None)
        if early_stopper.early_stop:
            break

    early_stopper.restore_best(latent_variable, cond if hparams.cond_key != "None" else None)
    return latent_variable, cond, early_stopper.best_loss


# =============================================================================
# 2. PROJECTION SINGLE PATCH (CON BENCHMARK E PLOT SUL CROP)
# =============================================================================
def project(
    ddim: DDIMSampler,
    decoder: torch.nn.Module,
    forward: ForwardDownsample,
    target: torch.Tensor,
    target_img_corrupted: torch.Tensor,
    device: torch.device,
    writer: SummaryWriter,
    hparams: Namespace,
    patch_stats: dict = None,
    verbose: bool = False,
):
    cat = {}
    crop_size = (hparams.crop_h, hparams.crop_w)

    if hparams.cond_key in ["concat", "hybrid", "crossattn"]:
        cat_path = Path(INPUT_FOLDER_CAT)
        with open(cat_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                obj_id = row["patch_id"]
                cat[obj_id] = {
                    "hi_size": float(row["hi_size"]),
                    "line_flux_integral": float(row["line_flux_integral"]),
                    "i": float(row["i"]),
                    "w20": float(row["w20"]),
                }

    spatial_mask = (target[:, :1] > 0).float() if target.shape[1] > 1 else (target > 0).float()
    _, latent_tmp = setup_noise_inputs(cat, device=device, hparams=hparams, batch_size=hparams.batch_size,)
    _, _, D_lat, H_lat, W_lat = latent_tmp.shape

    mask_latent = F.interpolate(
        spatial_mask, 
        size=(D_lat, H_lat, W_lat), 
        mode='trilinear', 
        align_corners=False
    ).detach() if hparams.cond_key in ["concat", "hybrid"] else None

    target_img_corrupted_crop = crop_center_3d(
        target_img_corrupted, (crop_size[0] // hparams.downsample_factor, crop_size[1] // hparams.downsample_factor)
    )
    target_crop = crop_center_3d(target, crop_size)

    vgg16, target_features = load_vgg_perceptual(hparams, target_img_corrupted_crop, device)
    if target_features is not None:
        target_features = target_features.detach()

    save_path = Path(hparams.output_dir_BRGM_ddim)
    save_path.mkdir(parents=True, exist_ok=True)

    target_crop_np = target_crop[0, 0].detach().cpu().numpy()
    target_phys_crop = denormalize_data(target_crop_np, norm_mode=hparams.norm_data, patch_stats=patch_stats)

    target_corrupted_crop_np = target_img_corrupted_crop[0, 0].detach().cpu().numpy()
    target_corrupted_phys_crop = denormalize_data(target_corrupted_crop_np, norm_mode=hparams.norm_data, patch_stats=patch_stats)

    logprint("\n[INFO] Esecuzione ottimizzazione del latente...", verbose)
    best_z, best_cond, best_loss = optimize_latent_patch(
        ddim=ddim,
        decoder=decoder,
        forward=forward,
        target_img_corrupted=target_img_corrupted,
        device=device,
        hparams=hparams,
        cat=cat,
        mask_latent=mask_latent,
        vgg16=vgg16,
        target_features=target_features,
        batch_size=hparams.batch_size
    )

    with torch.no_grad():
        cond_payload = {}
        if hparams.cond_key in ["concat", "hybrid"]:
            cond_payload["c_concat"] = [mask_latent]
        if hparams.cond_key in ["crossattn", "hybrid"]:
            context_attn = best_cond.unsqueeze(1) if (best_cond is not None and best_cond.ndim == 2) else best_cond
            cond_payload["c_crossattn"] = [context_attn]

        best_synth_img = sampling_from_ddim(
            ddim=ddim, 
            decoder=decoder, 
            latent_variable=best_z, 
            cond=cond_payload, 
            hparams=hparams
        )
        # NUOVO CODICE (Converte bfloat16 -> float32 prima di NumPy)
        synth_img_np = best_synth_img[0, 0].detach().float().cpu().numpy()
        
        synth_phys_full = denormalize_data(synth_img_np, norm_mode=hparams.norm_data, patch_stats=patch_stats)
        synth_phys_crop = crop_center_3d(synth_phys_full, crop_size)

    if verbose:
        step_str = f"{hparams.num_steps}".zfill(4)
        draw_img(target_phys_crop, title=f"target_{hparams.norm_data}", step=step_str, output_folder=save_path)
        compare_cubes(target_phys_crop, synth_phys_crop, title=f"target_vs_synth_{hparams.norm_data}", save_path=save_path / f"compare_target_vs_synth_{step_str}.png")
        draw_img(target_corrupted_phys_crop, title=f"corrupted_{hparams.norm_data}", step=step_str, output_folder=save_path)
        plot_orthogonal_cuts(synth_phys_crop, title=f"orthogonal_cuts_synth_{hparams.norm_data}", save_path=save_path / f"orthogonal_cuts_synth_{step_str}.png")

        fig = comparison_plots_ok(target_phys_crop, synth_phys_crop, x_lr=target_corrupted_phys_crop)
        fig.savefig(save_path / f"comparison_ok_{hparams.norm_data}_{step_str}.png")
        if writer is not None:
            writer.add_figure("Reconstruction_Crop", fig, global_step=hparams.num_steps)
        plt.close(fig)

        plot_zoomed_comparison(target_phys_crop, synth_phys_crop, save_path=save_path, filename="sr_zoomed_comparison_paper.png")

        k_vals, pk_sr_norm, pk_ref_norm, r_k, t_k = compute_power_spectrum_and_cross(synth_phys_crop, target_phys_crop)
        save_power_spectrum_plots(k_vals, pk_sr_norm, pk_ref_norm, r_k, t_k, save_path)

        benchmark_results = run_astrophysical_benchmarks(
            target_phys=target_phys_crop,
            synth_phys=synth_phys_crop,
            target_corrupted_phys=target_corrupted_phys_crop,
            latent_tensor=best_z,
            device=hparams.device
        )
        print("\n" + "="*50)
        print("         3D-SR / BRGM BENCHMARK RESULTS (CROP)   ")
        print("="*50)
        for key, val in benchmark_results.items():
            if "MB" in key or "%" in key or "Ratio" in key:
                print(f" {key:<35}: {val:.2f}")
            else:
                print(f" {key:<35}: {val:.6f}")
        print("="*50 + "\n")

        with open(save_path / "benchmark_report.json", "w") as f:
            json.dump(benchmark_results, f, indent=4)

    if writer is not None:
        writer.flush()
        writer.close()

    torch.save({
        "epoch": hparams.num_steps,
        "latent_variable": best_z,
        "cond": best_cond,
    }, save_path / "checkpoint.pth")

    return best_z, best_cond, {"loss": best_loss}


# =============================================================================
# 3. RECONSTRUCT LARGE CROP TILING
# =============================================================================
def reconstruct_large_crop_tiling(
    large_target_lr: torch.Tensor,
    large_target_hr: torch.Tensor,
    ddim: object,                  # DDIMSampler
    decoder: torch.nn.Module,
    forward: object,              # ForwardDownsample
    writer: SummaryWriter,
    patch_size: tuple = (128, 128),
    overlap: int = 32,
    batch_size: int = 1,           
    hparams: Namespace = None,
    device: torch.device = None,
    patch_stats: dict = None,
    verbose: bool = True
) -> torch.Tensor:
    
    _, _, D, H_large, W_large = large_target_hr.shape # 160 x 160
    pH, pW = patch_size # 128 x 128
    factor = hparams.downsample_factor

    stride_h, stride_w = pH - overlap, pW - overlap #128-32

    # 1. Pre-calcolo della maschera dei pesi base (windowing di Hann) 5D: [1, 1, 1, pH, pW]
    window_y = torch.hann_window(pH, periodic=False, device=device)
    window_x = torch.hann_window(pW, periodic=False, device=device)
    base_patch_weight = torch.outer(window_y, window_x).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    base_patch_weight = torch.clamp(base_patch_weight, min=1e-3) # 128 x 128

    # 2. Pre-calcolo di tutte le coordinate della griglia
    y_steps = list(range(0, H_large - pH + 1, stride_h)) # 0, 160-128+1=33, 96
    if y_steps[-1] + pH < H_large: 
        y_steps.append(H_large - pH) #160-128
    x_steps = list(range(0, W_large - pW + 1, stride_w))
    if x_steps[-1] + pW < W_large: 
        x_steps.append(W_large - pW) #160-128

    all_coords = [(y, x) for y in y_steps for x in x_steps]
    total_patches = len(all_coords)
    
    if verbose:
        print(f"\n[INFO] Tiling in batch attivo: {total_patches} sub-patch ({pH}x{pW}) raggruppati con batch_size={batch_size}...")

    # Pre-caricamento catalogo (una sola volta)
    cat = {}
    if hparams.cond_key in ["concat", "hybrid", "crossattn"]:
        cat_path = Path(INPUT_FOLDER_CAT)
        with open(cat_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                obj_id = row["patch_id"]
                cat[obj_id] = {
                    "hi_size": float(row["hi_size"]),
                    "line_flux_integral": float(row["line_flux_integral"]),
                    "i": float(row["i"]),
                    "w20": float(row["w20"]),
                }

    vgg16, _ = load_vgg_perceptual(hparams, large_target_lr, device)

    # Inizializzazioni differite (allocate una sola volta al primo batch)
    output_sr = None
    weight_map = None
    curr_patch_weight = None
    scale_h, scale_w = None, None

    has_concat = hparams.cond_key in ["concat", "hybrid"]
    has_crossattn = hparams.cond_key in ["crossattn", "hybrid"]

    # 3. CICLO DI ESECUZIONE PER BATCH
    for i in range(0, total_patches, batch_size):
        batch_coords = all_coords[i : i + batch_size]
        
        batch_hr_list = []
        batch_lr_list = []
        batch_mask_latent_list = []

        # Estrazione batch di sub-patch
        for y, x in batch_coords:
            p_hr = large_target_hr[:, :, :, y : y + pH, x : x + pW] #128x128
            
            y_lr, x_lr = y // factor, x // factor
            pH_lr, pW_lr = pH // factor, pW // factor
            p_lr = large_target_lr[:, :, :, y_lr : y_lr + pH_lr, x_lr : x_lr + pW_lr]

            if has_concat:
                spatial_mask = (p_hr[:, :1] > 0).float() if p_hr.shape[1] > 1 else (p_hr > 0).float()
                m_latent = F.interpolate(
                    spatial_mask, size=(D // 4, pH // 4, pW // 4), mode='trilinear', align_corners=False
                ).detach()
                batch_mask_latent_list.append(m_latent)

            batch_hr_list.append(p_hr)
            batch_lr_list.append(p_lr)

        batch_hr = torch.cat(batch_hr_list, dim=0)
        batch_lr = torch.cat(batch_lr_list, dim=0)
        batch_mask_latent = torch.cat(batch_mask_latent_list, dim=0) if batch_mask_latent_list else None

        patch_tag = f"Batch_{i // batch_size}"

        # Ottimizzazione latente in parallelo per l'intero batch
        best_z, best_cond, patch_best_loss = optimize_latent_patch(
            ddim=ddim,
            decoder=decoder,
            forward=forward,
            target_img_corrupted=batch_lr,
            device=device,
            hparams=hparams,
            patch_size=(128, patch_size[0], patch_size[1]),
            cat=cat,
            mask_latent=batch_mask_latent,
            vgg16=vgg16,
            writer=writer,
            batch_size=hparams.batch_size,
            patch_tag=patch_tag
        )

        if writer is not None:
            writer.add_scalar("Tiling/Batch_Final_Loss", patch_best_loss, global_step=i // batch_size)

        # Sampling DDIM per il batch
        with torch.no_grad():
            cond_payload = {}
            if has_concat:
                cond_payload["c_concat"] = [batch_mask_latent]
            if has_crossattn:
                context_attn = best_cond.unsqueeze(1) if (best_cond is not None and best_cond.ndim == 2) else best_cond
                cond_payload["c_crossattn"] = [context_attn]

            synth_patch_sr_batch = sampling_from_ddim(
                ddim=ddim, 
                decoder=decoder, 
                latent_variable=best_z, 
                cond=cond_payload, 
                hparams=hparams
            )                             #128x128

            if synth_patch_sr_batch.shape[0] != len(batch_coords):
                raise ValueError(
                    f"Mismatch nel batch: generati {synth_patch_sr_batch.shape[0]} patch "
                    f"ma inviati {len(batch_coords)} in batch_coords!"
                )
            
        # 4. ALLOCAZIONE E RIDIMENSIONAMENTO ESEGUITI UNA SOLA VOLTA AL PRIMO BATCH
        if output_sr is None:
            # Estraiamo le dimensioni reali dall'output della rete [B, C, D, H, W]
            _, C_out, D_out, pH_out, pW_out = synth_patch_sr_batch.shape # [1, C_out, D, 128, 128]
            
            scale_h = pH_out // pH # 128 // 128 = 1
            scale_w = pW_out // pW # 128 // 128 = 1

            H_target_out = H_large * scale_h # 160 * 1 = 160
            W_target_out = W_large * scale_w # 160 * 1 = 160

            
            # Canvas dell'immagine finale (ha C_out canali) -> [1, C_out, D, 160, 160]
            output_sr = torch.zeros((1, C_out, D_out, H_target_out, W_target_out), device=device)
            
            # Moppa dei pesi per la normalizzazione (ha SEMPRE 1 canale!) -> [1, 1, D, 160, 160]
            weight_map = torch.zeros((1, 1, D_out, H_target_out, W_target_out), device=device)

            # Ridimensionamento statico della maschera dei pesi se necessario
            if base_patch_weight.shape[-2:] != (pH_out, pW_out):
                curr_patch_weight = F.interpolate(
                    base_patch_weight.squeeze(2),
                    size=(pH_out, pW_out),
                    mode='bilinear',
                    align_corners=False
                ).unsqueeze(2)
            else:
                curr_patch_weight = base_patch_weight # Shape: [1, 1, 1, 128, 128]
        # Fai un test rapido:
        
        # 5. Accumulo sulla griglia globale per ogni patch nel batch
        for idx, (y, x) in enumerate(batch_coords):
            # Singola patch dal batch: [1, C_out, D_out, 128, 128]
            synth_patch_sr = synth_patch_sr_batch[idx].unsqueeze(0)
            y_out, x_out = y * scale_h, x * scale_w
            slice_canvas = output_sr[:, :, :, y_out : y_out + pH_out, x_out : x_out + pW_out]
            patch_weighted = synth_patch_sr * curr_patch_weight
            print(f"\n[BATCH LOOP] coords y={y}, x={x} | y_out={y_out}, x_out={x_out}")
            print(f" -> synth_patch_sr.shape:    {synth_patch_sr.shape}")
            print(f" -> curr_patch_weight.shape: {curr_patch_weight.shape}")
            print(f" -> patch_weighted.shape:    {patch_weighted.shape}")
            print(f" -> slice_canvas.shape:      {slice_canvas.shape}")
            # 1. Moltiplicazione patch * peso -> [1, C_out, D_out, 128, 128]
            patch_weighted = synth_patch_sr * curr_patch_weight

            # 2. Accumulo nel canvas [1, C_out, D_out, 160, 160] sulla fetta [128, 128]
            output_sr[:, :, :, y_out : y_out + pH_out, x_out : x_out + pW_out] += patch_weighted
            
            # 3. Accumulo pesi nella weight_map [1, 1, D_out, 160, 160] sulla fetta [128, 128]
            weight_map[:, :, :, y_out : y_out + pH_out, x_out : x_out + pW_out] += curr_patch_weight
        

    # Normalizzazione finale
    full_reconstructed_sr = output_sr / (weight_map + 1e-8)
    crop_size = (hparams.crop_h, hparams.crop_w)

    full_sr_np = full_reconstructed_sr[0, 0].detach().cpu().numpy()
    full_target_np = large_target_hr[0, 0].detach().cpu().numpy()
    full_lr_np = large_target_lr[0, 0].detach().cpu().numpy()

    full_sr_phys = denormalize_data(full_sr_np, norm_mode=hparams.norm_data, patch_stats=patch_stats)
    full_target_phys = denormalize_data(full_target_np, norm_mode=hparams.norm_data, patch_stats=patch_stats)
    full_lr_phys = denormalize_data(full_lr_np, norm_mode=hparams.norm_data, patch_stats=patch_stats)

    sr_crop = crop_center_3d(full_sr_phys, crop_size)
    target_crop = crop_center_3d(full_target_phys, crop_size)
    lr_crop = crop_center_3d(full_lr_phys, (crop_size[0] // factor, crop_size[1] // factor))

    save_path = Path(hparams.output_dir_BRGM_ddim)
    save_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n[INFO] Generazione plot e benchmark sul crop dell'immagine ricomposta...")
        fig = comparison_plots_ok(target_crop, sr_crop, x_lr=lr_crop)
        fig.savefig(save_path / f"tiling_reconstruction_crop_{hparams.norm_data}.png")
        
        if writer is not None:
            writer.add_figure("Tiling/Reconstruction_Crop", fig, global_step=0)
        plt.close(fig)

        plot_orthogonal_cuts(sr_crop, title=f"orthogonal_cuts_tiling_{hparams.norm_data}", save_path=save_path / "orthogonal_cuts_tiling.png")
        plot_zoomed_comparison(target_crop, sr_crop, save_path=save_path, filename="tiling_zoomed_comparison.png")

        k_vals, pk_sr_norm, pk_ref_norm, r_k, t_k = compute_power_spectrum_and_cross(sr_crop, target_crop)
        save_power_spectrum_plots(k_vals, pk_sr_norm, pk_ref_norm, r_k, t_k, save_path)

        benchmark_results = run_astrophysical_benchmarks(
            target_phys=target_crop,
            synth_phys=sr_crop,
            target_corrupted_phys=lr_crop,
            device=device
        )

        if writer is not None:
            for key, val in benchmark_results.items():
                if isinstance(val, (int, float)):
                    writer.add_scalar(f"Tiling_Metrics/{key}", val, global_step=0)

        with open(save_path / "tiling_benchmark_report.json", "w") as f:
            json.dump(benchmark_results, f, indent=4)

    if writer is not None:
        writer.flush()
        writer.close()

    return full_reconstructed_sr



# =============================================================================
# 4. MAIN ENTRY POINT
# =============================================================================
def main(hparams: Namespace) -> None:
    device = torch.device(hparams.device)
    factor = hparams.downsample_factor

    top_y = hparams.crop_top_y
    left_x = hparams.crop_left_x
    crop_h = hparams.crop_h
    crop_w = hparams.crop_w

    print(f"[INFO] Caricamento Crop HR ({crop_h}x{crop_w}) da: {hparams.hr_path}")
    target_hr, patch_stats = load_crop_from_path(
        file_path=hparams.hr_path,
        top_y=top_y,
        left_x=left_x,
        crop_h=crop_h,
        crop_w=crop_w,
        device=device,
        target_depth=128,
        norm_mode=hparams.norm_data
    )

    crop_h_lr = crop_h // factor
    crop_w_lr = crop_w // factor
    top_y_lr = top_y // factor
    left_x_lr = left_x // factor

    print(f"[INFO] Caricamento Crop LR ({crop_h_lr}x{crop_w_lr}) da: {hparams.lr_path}")
    target_lr, _ = load_crop_from_path(
        file_path=hparams.lr_path,
        top_y=top_y_lr,
        left_x=left_x_lr,
        crop_h=crop_h_lr,
        crop_w=crop_w_lr,
        device=device,
        target_depth=32,
        norm_mode=hparams.norm_data
    )

    diffusion, decoder = load_pre_trained_model(hparams, device=device)
    
    ddim = DDIMSampler(diffusion)
    
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        decoder = decoder.to(dtype=torch.bfloat16)
    else:
        decoder = decoder.to(dtype=torch.float16)

    # Compile after casting
    decoder = torch.compile(decoder, mode="reduce-overhead")

    forward = create_corruption_function(hparams=hparams, device=device)
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    TB_LOG_DIR = hparams.tensor_board_logger_ddim
    log_dir = f"{TB_LOG_DIR}/run_{current_time}"
    writer = SummaryWriter(log_dir=log_dir)
    if crop_h == 128 and crop_w == 128:
        final_z, final_cond, _ = project(
            ddim=ddim,
            decoder=decoder,
            forward=forward,
            target=target_hr,
            target_img_corrupted=target_lr,
            device=device,
            writer=writer,
            hparams=hparams,
            patch_stats=patch_stats,
            verbose=True
        )
        save_path = hparams.output_dir_BRGM_ddim
        torch.save({"z": final_z, "cond": final_cond}, f"{save_path}/results.pth")
        print(f"[INFO] Risultati salvati in: {save_path}")
    else:
        final_large_sr = reconstruct_large_crop_tiling(
            large_target_lr=target_lr,
            large_target_hr=target_hr,
            ddim=ddim,
            decoder=decoder,
            forward=forward,
            writer=writer,
            patch_size=(128, 128),
            overlap=32,
            hparams=hparams,
            device=device,
            patch_stats=patch_stats,
            verbose=True
        )
        save_path = Path(hparams.output_dir_BRGM_ddim)
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / "large_reconstructed_sr.npy", final_large_sr.cpu().numpy())
        print(f"[INFO] Cubo grande SR salvato con successo in: {save_path / 'large_reconstructed_sr.npy'}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Diffusion Model Inversion for Astrophysical Data")
    add_argument(parser)
    args = parser.parse_args()
    main(args)