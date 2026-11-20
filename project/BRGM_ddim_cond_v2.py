# Code adapted for Astrophysical Data Restoration
# Original Reference: Pinaya et al. (2022) & Marinescu et al. (2020)

import csv
import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models.BRGM.forward_models import ForwardDownsample
from models.ddim import DDIMSampler
from utils.add_argument import add_argument
from utils.const import INPUT_FOLDER_CAT
from utils.plot_new import (
    compare_cubes,
    comparison_plots_ok,
    denormalize_data,
    draw_img,
    plot_orthogonal_cuts,
)
from utils.utils_new import (
    create_corruption_function,
    getVggFeatures,
    load_pre_trained_model,
    load_target_image,
    load_vgg_perceptual,
    sampling_from_ddim,
    setup_noise_inputs,
    normalize_dynamic,
)

from utils.power_spectrum_new import compute_power_spectrum_and_cross, save_power_spectrum_plots
from utils.metrics_benchmark import run_astrophysical_benchmarks
from utils.early_stop import LatentEarlyStopping


def plot_zoomed_comparison(
    target_phys: np.ndarray,
    synth_phys: np.ndarray,
    save_path: Path,
    crop_size_ratio: float = 0.35,
    filename: str = "sr_zoomed_comparison_paper.png",
) -> None:
    """
    Generates a single high-quality, publication-ready figure comparing HR target
    and reconstructed SR outputs, with an automatically centered zoomed ROI.
    """
    # Extract 2D slice with peak signal if input is 3D cube
    if target_phys.ndim == 3:
        max_idx = np.unravel_index(np.argmax(target_phys), target_phys.shape)
        slice_idx = max_idx[0]
        target_2d = target_phys[slice_idx]
        synth_2d = synth_phys[slice_idx]
    else:
        target_2d = target_phys
        synth_2d = synth_phys

    h, w = target_2d.shape

    # Find peak position to center ROI
    max_y, max_x = np.unravel_index(np.argmax(target_2d), (h, w))

    crop_h = int(h * crop_size_ratio)
    crop_w = int(w * crop_size_ratio)

    # Ensure box remains within image boundaries
    y1 = max(0, min(h - crop_h, max_y - crop_h // 2))
    y2 = min(h, y1 + crop_h)
    x1 = max(0, min(w - crop_w, max_x - crop_w // 2))
    x2 = min(w, x1 + crop_w)

    target_crop = target_2d[y1:y2, x1:x2]
    synth_crop = synth_2d[y1:y2, x1:x2]

    vmin = min(target_2d.min(), synth_2d.min())
    vmax = max(target_2d.max(), synth_2d.max())

    # Publication style settings
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.0), dpi=300)

    # Top-Left: Full HR Target
    im0 = axes[0, 0].imshow(target_2d, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Ground Truth (Target HR)", fontweight='bold')
    axes[0, 0].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.8, edgecolor='cyan', facecolor='none', linestyle='--'))
    axes[0, 0].set_xlabel("X [pixels]")
    axes[0, 0].set_ylabel("Y [pixels]")

    # Top-Right: Full SR Reconstructed
    axes[0, 1].imshow(synth_2d, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Reconstructed (SR Model)", fontweight='bold')
    axes[0, 1].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.8, edgecolor='cyan', facecolor='none', linestyle='--'))
    axes[0, 1].set_xlabel("X [pixels]")
    axes[0, 1].set_ylabel("Y [pixels]")

    # Bottom-Left: Zoomed HR Target
    axes[1, 0].imshow(target_crop, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f"Target Zoom (ROI: [{x1}:{x2}, {y1}:{y2}])", fontweight='bold')
    axes[1, 0].set_xlabel("X [pixels]")
    axes[1, 0].set_ylabel("Y [pixels]")

    # Bottom-Right: Zoomed SR Reconstructed
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


def load_lr_patch_from_path(
    lr_path: str,
    device: torch.device,
    norm_mode: str = "minmax",
    global_stats: dict = None,
) -> torch.Tensor:
    if lr_path.endswith(".npy"):
        data = np.load(lr_path).astype(np.float32)
        lr_tensor = torch.from_numpy(data)
    elif lr_path.endswith((".pt", ".pth")):
        lr_tensor = torch.load(lr_path, map_location=device)
    else:
        raise ValueError(f"Unsupported file format: {lr_path}")

    # Ensure shape is (1, C, D, H, W)
    if lr_tensor.ndim == 3:
        lr_tensor = lr_tensor.unsqueeze(0).unsqueeze(0)
    elif lr_tensor.ndim == 4:
        lr_tensor = lr_tensor.unsqueeze(0)

    lr_tensor = lr_tensor.to(device)

    # Apply the same normalization as target
    if global_stats is not None:
        clamp_min = -1.0 if norm_mode == "zscore" else 0.0
        lr_tensor, _ = normalize_dynamic(
            lr_tensor, norm_mode=norm_mode, stats=global_stats
        )
        lr_tensor = torch.clamp(lr_tensor, clamp_min, 1.0)

    return lr_tensor


def denormalize_cond(cond: torch.Tensor, catalogue: pd.DataFrame, feature_cols: list) -> torch.Tensor:
    """Denormalizes parameters based on physical values from the catalogue."""
    mins = [catalogue[col].min() for col in feature_cols]
    maxs = [catalogue[col].max() for col in feature_cols]
    
    stats_min = torch.tensor(mins, device=cond.device, dtype=torch.float32)
    stats_max = torch.tensor(maxs, device=cond.device, dtype=torch.float32)
    
    current_cond_phys = cond.detach() * (stats_max - stats_min) + stats_min
    return torch.clamp(current_cond_phys, stats_min, stats_max)


def logprint(message: str, verbose: bool) -> None:
    if verbose:
        print(message)


def create_mask_for_backprop(hparams: Namespace, device: torch.device) -> torch.Tensor:
    mask_cond = torch.ones((1, 4), device=device)
    mask_cond[:, 0] = 0 if not hparams.update_hi_size else 1
    mask_cond[:, 1] = 0 if not hparams.update_line_flux_integral else 1
    mask_cond[:, 2] = 0 if not hparams.update_i else 1
    mask_cond[:, 3] = 0 if not hparams.update_w20 else 1
    return mask_cond


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
    # 1. CATALOGUE INITIAL SETUP & CONDITIONING
    cat = {}
    update_params = []
    cond = None
    mask_latent = None
    cond_out = None
    
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
        feature_cols = ['hi_size', 'line_flux_integral', 'i', 'w20']
        cat_df = pd.DataFrame.from_dict(cat, orient='index')
    
        # Initialize cond and latent_variable
        cond, latent_variable = setup_noise_inputs(cat, device=device, hparams=hparams, batch_size=hparams.batch_size)
        
        # SPATIAL MASK PREPARATION FOR CONCAT CONDITIONING
        spatial_mask = (target[:, :1] > 0).float() if target.shape[1] > 1 else (target > 0).float()
        _, _, D_lat, H_lat, W_lat = latent_variable.shape

        mask_latent = F.interpolate(
            spatial_mask, 
            size=(D_lat, H_lat, W_lat), 
            mode='trilinear', 
            align_corners=False
        ).detach()

        if getattr(hparams, "update_conditioning", False):
            cond = cond.detach().requires_grad_(True)
            update_params.append(cond)

        cond_out = torch.zeros(
            [hparams.num_steps] + list(cond.shape[1:]),
            dtype=torch.float32,
            device=device,
        )
    else:
        _, latent_variable = setup_noise_inputs(cat, device=device, hparams=hparams)

    # 2. LATENT OPTIMIZATION SETUP
    if getattr(hparams, "update_latent_variables", True):
        latent_variable = latent_variable.detach().requires_grad_(True)
        update_params.append(latent_variable)

    optimizer = None
    scheduler = None
    if len(update_params) > 0:
        optimizer = torch.optim.Adam(
            update_params, 
            betas=(0.9, 0.999), 
            lr=hparams.learning_rate
        )

        # SCHEDULER INITIALIZATION
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=hparams.num_steps - hparams.start_steps, 
            eta_min=1e-6
        )

    latent_variable_out = torch.zeros(
        [hparams.num_steps] + list(latent_variable.shape[1:]),
        dtype=torch.float32,
        device=device,
    )

    # 3. CORRUPTED TARGET & VGG PREPARATION
    vgg16, target_features = load_vgg_perceptual(hparams, target_img_corrupted, device)
    if target_features is not None:
        target_features = target_features.detach()

    save_path = Path(hparams.output_dir_BRGM_ddim)
    save_path.mkdir(parents=True, exist_ok=True)

    latest_metrics = {}
    target_np = target[0, 0].detach().cpu().numpy()
    target_phys = denormalize_data(target_np, norm_mode=hparams.norm_data, patch_stats=patch_stats)

    # 4. EARLY STOPPER INITIALIZATION
    early_stopper = LatentEarlyStopping(
        patience=hparams.patience,
        min_delta=hparams.min_delta,
        save_to_disk=False  # Mantiene i latenti migliori in RAM per massima velocità
    )

    # 5. OPTIMIZATION LOOP
    for step in range(hparams.start_steps, hparams.num_steps):
        step_synth_img = None
        step_synth_img_corrupted = None
        step_pixel_loss = None
        step_perc_loss = None
        
        def closure():
            nonlocal step_synth_img, step_synth_img_corrupted, step_pixel_loss, step_perc_loss
            if optimizer is not None:
                optimizer.zero_grad()

            # A. CONDITIONING PAYLOAD
            cond_payload = {}
            if hparams.cond_key in ["concat", "hybrid"]:
                cond_payload["c_concat"] = [mask_latent]

            if hparams.cond_key in ["crossattn", "hybrid"]:
                context_attn = cond.unsqueeze(1) if cond.ndim == 2 else cond
                cond_payload["c_crossattn"] = [context_attn]

            # B. GENERATION
            synth_img = sampling_from_ddim(
                ddim=ddim,
                decoder=decoder,
                latent_variable=latent_variable,
                cond=cond_payload,
                hparams=hparams,
            )

            # C. CORRUPTION AND WEIGHTED LOSS
            synth_img_corrupted = forward(synth_img)
            weights = torch.abs(target_img_corrupted) + 1.0
            pixel_loss = ((synth_img_corrupted - target_img_corrupted).abs() * weights).mean()
            loss = pixel_loss

            # Prior Loss
            prior_loss = torch.tensor(0.0, device=device)
            if hasattr(hparams, "lambda_prior") and hparams.lambda_prior > 0:
                prior_loss = torch.mean(latent_variable ** 2)
                loss += hparams.lambda_prior * prior_loss

            # Perceptual Loss
            perc_loss = torch.tensor(0.0, device=device)
            if hparams.lambda_perc > 0 and vgg16 is not None:
                synth_features = getVggFeatures(hparams, synth_img_corrupted, vgg16)
                perc_loss = (target_features - synth_features).abs().mean()
                loss += hparams.lambda_perc * perc_loss

            # D. BACKPROPAGATION
            if len(update_params) > 0 and loss.requires_grad:
                loss.backward()
                if getattr(hparams, "update_conditioning", False) and cond is not None and cond.grad is not None:
                    mask_cond = create_mask_for_backprop(hparams, device)
                    cond.grad *= mask_cond

            step_synth_img = synth_img
            step_synth_img_corrupted = synth_img_corrupted
            step_pixel_loss = pixel_loss.item()
            step_perc_loss = perc_loss.item()
            
            return loss

        # EXECUTE OPTIMIZATION STEP
        if optimizer is not None and step % hparams.prior_every == 0:
            loss_tensor = optimizer.step(closure=closure)
            current_loss = loss_tensor.item()
        else:
            with torch.no_grad():
                closure()
                current_loss = (step_pixel_loss or 0.0) + (step_perc_loss or 0.0)

        # SCHEDULER STEP
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else hparams.learning_rate
        if scheduler is not None:
            scheduler.step()

        # CLAMP CONDITIONING
        if hparams.cond_key != "None" and cond is not None:
            with torch.no_grad():
                cond.clamp_(0, 1)
            cond_out[step] = cond.detach()[0]
            
        latent_variable_out[step] = latent_variable.detach()[0]

        # CONTROL EARLY STOPPING
        early_stopper(current_loss, latent_variable, cond if hparams.cond_key != "None" else None)

        # METRICS CALCULATION & TENSORBOARD LOGGING
        with torch.no_grad():
            synth_img_np = step_synth_img[0, 0].detach().cpu().numpy()
            synth_phys = denormalize_data(synth_img_np, norm_mode=hparams.norm_data, patch_stats=patch_stats)

            ssim_range = 1.0 if hparams.norm_data != 'zscore' else 2.0
            ssim_ = ssim(synth_phys, target_phys, win_size=11, data_range=ssim_range, gaussian_weights=True, use_sample_covariance=False)

            global_max = max(target_phys.max(), synth_phys.max())
            global_min = min(target_phys.min(), synth_phys.min())
            data_range = global_max - global_min if global_max > global_min else 1e-5

            psnr_ = psnr(target_phys, synth_phys, data_range=data_range)
            mse_ = mse(target_phys, synth_phys)
            nmse_ = nmse(target_phys, synth_phys)

            if hparams.cond_key != "None" and cond is not None:
                cond_phys = denormalize_cond(cond, catalogue=cat_df, feature_cols=feature_cols)
                if verbose:
                    print(f"Step {step:03d} | LR: {current_lr:.2e} | Loss: {current_loss:.6f} | Hi Size: {cond_phys[0,0]:.4f} | Line Flux: {cond_phys[0,1]:.4f} | I: {cond_phys[0,2]:.4f} | W20: {cond_phys[0,3]:.4f} | SSIM: {ssim_:.4f}")
            else:
                if verbose:
                    print(f"Step {step:03d} | LR: {current_lr:.2e} | Loss: {current_loss:.6f} | SSIM: {ssim_:.4f}")

            writer.add_scalar("Loss/Total", current_loss, global_step=step)
            writer.add_scalar("Loss/Pixelwise", step_pixel_loss, global_step=step)
            writer.add_scalar("Loss/Perceptual", step_perc_loss, global_step=step)
            writer.add_scalar("Params/LearningRate", current_lr, global_step=step)
            writer.add_scalar("Metrics/SSIM", ssim_, global_step=step)
            writer.add_scalar("Metrics/PSNR", psnr_, global_step=step)
            writer.add_scalar("Metrics/MSE", mse_, global_step=step)
            writer.add_scalar("Metrics/NMSE", nmse_, global_step=step)

            # PLOT EXPORT (Every N Steps)
            if step % 50 == 0 or step == hparams.num_steps - 1:
                target_img_corrupted_phys = denormalize_data(target_img_corrupted[0, 0].detach().cpu().numpy(), norm_mode=hparams.norm_data, patch_stats=patch_stats)
                step_str = f"{step}".zfill(4)                      
                
                fig = comparison_plots_ok(target_phys, synth_phys, x_lr=target_img_corrupted_phys)
                fig.savefig(save_path / f"comparison_ok_{hparams.norm_data}_{step_str}.png")                
                
                writer.add_figure("Reconstruction", fig, global_step=step)
                plt.close(fig)

            latest_metrics = {"loss": current_loss, "ssim": ssim_, "psnr": psnr_, "mse": mse_, "nmse": nmse_}

        if early_stopper.early_stop:
            logprint(f"\n[INFO] Early Stopping attivato allo step {step}!", verbose)
            break

    writer.flush()
    writer.close()

    # 6. RESTORE BEST LATENTS & RE-GENERATE BEST OUTPUT
    logprint("\n[INFO] Ripristino dei migliori latenti ottenuti durante l'ottimizzazione...", verbose)
    early_stopper.restore_best(latent_variable, cond if hparams.cond_key != "None" else None)

    with torch.no_grad():
        cond_payload = {}
        if hparams.cond_key in ["concat", "hybrid"]:
            cond_payload["c_concat"] = [mask_latent]
        if hparams.cond_key in ["crossattn", "hybrid"]:
            context_attn = cond.unsqueeze(1) if cond.ndim == 2 else cond
            cond_payload["c_crossattn"] = [context_attn]

        best_synth_img = sampling_from_ddim(
            ddim=ddim,
            decoder=decoder,
            latent_variable=latent_variable,
            cond=cond_payload,
            hparams=hparams,
        )
        best_synth_img_corrupted = forward(best_synth_img)

        synth_img_np = best_synth_img[0, 0].detach().cpu().numpy()
        synth_phys = denormalize_data(synth_img_np, norm_mode=hparams.norm_data, patch_stats=patch_stats)
        synth_img_corrupted_phys = denormalize_data(best_synth_img_corrupted[0, 0].detach().cpu().numpy(), norm_mode=hparams.norm_data, patch_stats=patch_stats)

    # 7. FINAL PLOTS & EXPORTS
    step_str = f"{step}".zfill(4)
    draw_img(target_phys, title=f"target_{hparams.norm_data}", step=step_str, output_folder=save_path)
    compare_cubes(target_phys, synth_phys, title=f"target_vs_synth_{hparams.norm_data}", save_path=save_path / f"compare_target_vs_synth_{step_str}.png")
    draw_img(synth_img_corrupted_phys, title=f"corrupted_{hparams.norm_data}", step=step_str, output_folder=save_path)
    plot_orthogonal_cuts(synth_phys, title=f"orthogonal_cuts_synth_{hparams.norm_data}", save_path=save_path / f"orthogonal_cuts_synth_{step_str}.png")

    # HIGH-QUALITY PAPER ZOOM COMPARISON PLOT
    logprint("[INFO] Generating high-resolution zoomed ROI comparison plot for publication...", verbose)
    plot_zoomed_comparison(target_phys, synth_phys, save_path=save_path, filename="sr_zoomed_comparison_paper.png")

    # POWER SPECTRUM
    logprint("[INFO] Computing 3D Power Spectrum and Cross-Spectrum...", verbose)
    k_vals, pk_sr_norm, pk_ref_norm, r_k, t_k = compute_power_spectrum_and_cross(
        synth_phys,
        target_phys        
    )
    save_power_spectrum_plots(k_vals, pk_sr_norm, pk_ref_norm, r_k, t_k, save_path)

    # BENCHMARKS
    logprint("[INFO] Computing advanced benchmarks...", verbose)
    target_corrupted_phys_final = denormalize_data(target_img_corrupted[0, 0].detach().cpu().numpy(), norm_mode=hparams.norm_data, patch_stats=patch_stats)

    benchmark_results = run_astrophysical_benchmarks(
        target_phys=target_phys,
        synth_phys=synth_phys,
        target_corrupted_phys=target_corrupted_phys_final,
        latent_tensor=latent_variable,
        device=hparams.device
    )

    if verbose:
        print("\n" + "="*50)
        print("         3D-SR / BRGM BENCHMARK RESULTS         ")
        print("="*50)
        for key, val in benchmark_results.items():
            if "MB" in key or "%" in key or "Ratio" in key:
                print(f" {key:<35}: {val:.2f}")
            else:
                print(f" {key:<35}: {val:.6f}")
        print("="*50 + "\n")

    with open(save_path / "benchmark_report.json", "w") as f:
        json.dump(benchmark_results, f, indent=4)

    os.makedirs(hparams.output_dir_BRGM_ddim, exist_ok=True)
    
    checkpoint_dict = {
        "epoch": step,
        "latent_variable": latent_variable,
        "cond": cond,
    }
    if optimizer is not None:
        checkpoint_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint_dict["scheduler"] = scheduler.state_dict()

    torch.save(checkpoint_dict, f"{hparams.output_dir_BRGM_ddim}/checkpoint.pth")

    return latent_variable_out, cond_out, {"loss": early_stopper.best_loss, "ssim": latest_metrics.get("ssim", 0.0)}


def main(hparams: Namespace) -> None:
    device = torch.device(hparams.device)
    
    target_file = hparams.hr_path
    lr_file_path = hparams.lr_path
    img_tensor, patch_stats = load_target_image(hparams, device=device, target_path=target_file)
    if img_tensor.ndim == 4:
        img_tensor = img_tensor.unsqueeze(0)

    if os.path.exists(lr_file_path):
        target_lr = load_lr_patch_from_path(lr_file_path, device=device, norm_mode=hparams.norm_data, global_stats=patch_stats)
        print(f"[INFO] LR patch loaded successfully from: {lr_file_path}")
    else:
        raise FileNotFoundError(f"Unable to find LR file at path: {lr_file_path}")

    diffusion, decoder = load_pre_trained_model(hparams, device=device)
    ddim = DDIMSampler(diffusion)
    forward = create_corruption_function(hparams=hparams, device=device)
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    TB_LOG_DIR = hparams.tensor_board_logger_ddim
    log_dir = f"{TB_LOG_DIR}/run_{current_time}"
    writer = SummaryWriter(log_dir=log_dir)
    file_names = [target_file, Path(lr_file_path).name]
    final_z, final_cond, _ = project(
        ddim=ddim,
        decoder=decoder,
        forward=forward,
        target=img_tensor,
        target_img_corrupted=target_lr,
        device=device,
        writer=writer,
        hparams=hparams,
        patch_stats=patch_stats,
        verbose=True,
    )

    save_path = hparams.output_dir_BRGM_ddim
    torch.save({"z": final_z, "cond": final_cond}, f"{save_path}/results.pth")
    print(f"Results saved in {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Diffusion Model Inversion for Astrophysical Data")
    add_argument(parser)
    args = parser.parse_args()
    main(args)