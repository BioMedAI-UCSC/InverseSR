# Code is adpated from: https://huggingface.co/spaces/Warvito/diffusion_brain/blob/main/app.py and
# https://colab.research.google.com/drive/1xJAor6_Ky36gxIk6ICNP--NMBjSlYKil?usp=sharing#scrollTo=4XDeCy-Vj59b
# A lot of thanks to the author of the code
# Reference:
# [1] Pinaya, W. H., et al. (2022). "Brain Imaging Generation with Latent Diffusion Models." arXiv preprint arXiv:2209.07162.
# [2] Marinescu, R., et al. (2020). Bayesian Image Reconstruction using Deep Generative Models.

# TODO: remember to check for the tensorboard

import math

# from joblib import dump, load
from argparse import ArgumentParser, Namespace
from time import perf_counter
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
from models.BRGM.forward_models import ForwardAbstract
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.neighbors import NearestNeighbors
from torch.utils.tensorboard import SummaryWriter

from utils.add_argument import add_argument
from utils.const import (
    INPUT_FOLDER,
    LATENT_SHAPE,
    OUTPUT_FOLDER,
    PRETRAINED_MODEL_DDPM_PATH,
    PRETRAINED_MODEL_DECODER_PATH,
)
from utils.plot import draw_corrupted_images, draw_images, draw_img
from utils.utils import (
    create_corruption_function,
    generating_latent_vector,
    getVggFeatures,
    inference,
    load_ddpm_latent_vectors,
    load_ddpm_model,
    load_pre_trained_decoder,
    load_target_image,
    load_vgg_perceptual,
    setup_noise_inputs,
)


def logprint(message: str, verbose: bool) -> None:
    if verbose:
        print(message)


def add_hparams_to_tensorboard(
    hparams: Namespace,
    final_loss: torch.Tensor,
    final_ssim: float,
    final_psnr: float,
    final_mse: float,
    final_nmse: float,
    inversed_gender: float,
    inversed_age: float,
    inversed_ventricular: float,
    inversed_brain: float,
    writer: SummaryWriter,
) -> None:
    writer.add_hparams(
        {
            "num_steps": hparams.num_steps,
            "learning_rate": hparams.learning_rate,
            "experiment_name": hparams.experiment_name,
            "subject_id": hparams.subject_id,
            "update_latent_variables": hparams.update_latent_variables,
            "update_conditioning": hparams.update_conditioning,
            "update_gender": hparams.update_gender,
            "update_age": hparams.update_age,
            "update_ventricular": hparams.update_ventricular,
            "update_brain": hparams.update_brain,
            "alpha": hparams.lambda_alpha,
            "perc": hparams.lambda_perc,
            "kernel_size": hparams.kernel_size,
        },
        {
            "loss/final_loss": final_loss,
            "measurement/final_ssim": final_ssim,
            "measurement/final_psnr": final_psnr,
            "measurement/final_mse": final_mse,
            "measurement/final_nmse": final_nmse,
            "conditional_variable/inversed_gender": inversed_gender,
            "conditional_variable/inversed_age": inversed_age,
            "conditional_variable/inversed_ventricular": inversed_ventricular,
            "conditional_variable/inversed_brain": inversed_brain,
        },
    )


def create_mask_for_backprop(hparams: Namespace, device: torch.device) -> torch.Tensor:
    mask_cond = torch.ones((1, 4), device=device)
    mask_cond[:, 0] = 0 if not hparams.update_gender else 1
    mask_cond[:, 1] = 0 if not hparams.update_age else 1
    mask_cond[:, 2] = 0 if not hparams.update_ventricular else 1
    mask_cond[:, 3] = 0 if not hparams.update_brain else 1
    return mask_cond


def compute_latent_vector_stats(
    latent_vectors: torch.Tensor,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logprint("Computing latent vector stats", verbose)
    latent_mean = torch.mean(latent_vectors, axis=0, keepdim=True)
    latnet_std = torch.std(latent_vectors, dim=0, keepdim=True, unbiased=False)
    return latent_mean, latnet_std


def compute_prior_loss(
    cur_latent_vector: torch.Tensor,
    latent_vectors: torch.Tensor,
    latent_vector_std: torch.Tensor,
    knn_model: NearestNeighbors,
    hparams: Namespace,
) -> Tuple[torch.Tensor, List[int]]:
    cur_latent_vector_np = cur_latent_vector.detach().cpu().numpy().reshape((1, -1))
    _, indices = knn_model.kneighbors(cur_latent_vector_np, n_neighbors=hparams.k)
    nearest_latent_vectors = latent_vectors[indices[0]]
    mean_nearest_latent_vector = torch.mean(
        nearest_latent_vectors, axis=0, keepdim=True
    )
    prior_loss = (
        (
            (cur_latent_vector / latent_vector_std)
            - (mean_nearest_latent_vector / latent_vector_std)
        )
        .abs()
        .mean()
    )
    return prior_loss, indices[0]


def project(
    vqvae: torch.nn.Module,
    forward: ForwardAbstract,  # Corruption function
    target: torch.Tensor,
    device: torch.device,
    writer: SummaryWriter,
    hparams: Namespace,
    verbose: bool = False,
):
    latent_vectors_tensor = load_ddpm_latent_vectors(device)
    latent_vector_mean, latent_vector_std = compute_latent_vector_stats(
        latent_vectors=latent_vectors_tensor, device=device, verbose=verbose
    )

    if not hparams.mean_latent_vector:
        ddpm = load_ddpm_model(ddpm_path=PRETRAINED_MODEL_DDPM_PATH, device=device)
        cond, latent_variable = setup_noise_inputs(device=device)
        cond_crossatten = cond.unsqueeze(1)
        cond_concat = cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        cond_concat = cond_concat.expand(list(cond.shape[0:2]) + list(LATENT_SHAPE[2:]))
        conditioning = {
            "c_concat": [cond_concat.float().to(device)],
            "c_crossattn": [cond_crossatten.float().to(device)],
        }
        latent_vector = generating_latent_vector(
            diffusion=ddpm,
            latent_variable=latent_variable,
            conditioning=conditioning,
            batch_size=1,
        )
    else:
        latent_vector = latent_vector_mean.clone().detach()
    latent_vector.requires_grad = True

    update_params = []
    update_params.append(latent_vector)

    optimizer_adam = torch.optim.Adam(
        update_params,
        betas=(0.9, 0.999),
        lr=hparams.learning_rate,
    )
    latent_vector_out = torch.zeros(
        [hparams.num_steps] + list(latent_vector.shape[1:]),
        dtype=torch.float32,
        device=device,
    )

    target_img_corrupted = forward(target)
    vgg16, target_features = load_vgg_perceptual(hparams, target_img_corrupted, device)
    total_num_pixels = (
        target_img_corrupted.numel()
        if hparams.corruption != "mask"
        else math.prod(forward.mask.shape) - forward.mask.sum()
    )

    # Compute latent representation stats.
    for step in range(hparams.start_steps, hparams.num_steps):

        def closure():
            optimizer_adam.zero_grad()

            synth_img = inference(
                vqvae=vqvae,
                latent_vectors=latent_vector,
            )
            synth_img_corrupted = forward(synth_img)

            loss = 0
            downsampling_loss = 0
            prior_loss = 0
            indices = [0]
            if hparams.corruption != "None":
                pixelwise_loss = (
                    synth_img_corrupted - target_img_corrupted
                ).abs().sum() / total_num_pixels
                loss += pixelwise_loss

                synth_features = getVggFeatures(hparams, synth_img_corrupted, vgg16)
                perc_loss = (target_features - synth_features).abs().mean()
                loss += hparams.lambda_perc * perc_loss
            else:
                pixelwise_loss = (synth_img - target).abs().mean()
                loss += (1 - hparams.alpha_downsampling_loss) * pixelwise_loss

                synth_features = getVggFeatures(hparams, synth_img_corrupted, vgg16)
                perc_loss = (target_features - synth_features).abs().mean()
                loss += hparams.lambda_perc * perc_loss

                if hparams.downsampling_loss:
                    downsampling_synth_img = F.interpolate(
                        synth_img,
                        scale_factor=1 / hparams.downsampling_loss_factor,
                        mode="trilinear",
                    )
                    downsampling_loss = (
                        (downsampling_synth_img - downsampling_target_img).abs().mean()
                    )
                    loss += hparams.alpha_downsampling_loss * downsampling_loss

                # Don't use prior loss for there is no corruption.

            loss.backward(create_graph=False)

            return (
                loss,
                pixelwise_loss,
                perc_loss,
                downsampling_loss,
                prior_loss,
                synth_img,
                synth_img_corrupted,
                indices,
            )

        (
            loss,
            pixelwise_loss,
            perc_loss,
            downsampling_loss,
            prior_loss,
            synth_img,
            synth_img_corrupted,
            indices,
        ) = optimizer_adam.step(closure=closure)

        synth_img_np = synth_img[0, 0].detach().cpu().numpy()
        target_np = target[0, 0].detach().cpu().numpy()
        ssim_ = ssim(
            synth_img_np,
            target_np,
            win_size=11,
            data_range=1.0,
            gaussian_weights=True,
            use_sample_covariance=False,
        )
        # Code for computing PSNR is adapted from
        # https://github.com/agis85/multimodal_brain_synthesis/blob/master/error_metrics.py#L32
        data_range = np.max([synth_img_np.max(), target_np.max()]) - np.min(
            [synth_img_np.min(), target_np.min()]
        )
        psnr_ = psnr(target_np, synth_img_np, data_range=data_range)
        mse_ = mse(target_np, synth_img_np)
        nmse_ = nmse(target_np, synth_img_np)

        writer.add_scalar("loss", loss, global_step=step)
        writer.add_scalar("pixelwise_loss", pixelwise_loss, global_step=step)
        writer.add_scalar("perceptual_loss", perc_loss, global_step=step)
        writer.add_scalar("downsampling_loss", downsampling_loss, global_step=step)
        if prior_loss != 0:
            writer.add_scalar("prior_loss", prior_loss, global_step=step)
            writer.add_scalar("indice", indices[0], global_step=step)
        writer.add_scalar("ssim", ssim_, global_step=step)
        writer.add_scalar("psnr", psnr_, global_step=step)
        writer.add_scalar("mse", mse_, global_step=step)
        writer.add_scalar("nmse", nmse_, global_step=step)

        logprint(
            f"step {step + 1:>4d}/{hparams.num_steps}: tloss {float(loss):<5.8f} pix_loss {float(pixelwise_loss):<5.8f} perc_loss {float(perc_loss):<1.15f} prior_loss {float(prior_loss):<5.8f}\n"
            f"              : SSIM {float(ssim_):<5.8f} PSNR {float(psnr_):<5.8f} MSE {float(mse_):<5.8f} NMSE {float(nmse_):<5.8f}",
            verbose=verbose,
        )

        step_ = f"{step}".zfill(4)
        draw_img(
            synth_img_np,
            title="synth",
            step=step_,
            output_folder=OUTPUT_FOLDER,
        )

        if step % 25 == 0:
            if hparams.corruption != "None":
                imgs = draw_corrupted_images(
                    synth_img_np,
                    target_np,
                    synth_img_corrupted[0, 0].detach().cpu().numpy(),
                    target_img_corrupted[0, 0].detach().cpu().numpy(),
                    ssim_=ssim_,
                )
            else:
                imgs = draw_images(
                    synth_img_np,
                    target_np,
                    ssim_=ssim_,
                )
            step_ = f"{step}".zfill(4)
            writer.add_figure(f"step: {step_}", imgs, global_step=step)
            plt.close(imgs)

        latent_vector_out[step] = latent_vector.detach()[0]

    add_hparams_to_tensorboard(
        hparams,
        final_loss=loss.item(),
        final_ssim=ssim_,
        final_psnr=psnr_,
        final_mse=mse_,
        final_nmse=nmse_,
        inversed_gender=cond[0, 0].clone().detach().item(),
        inversed_age=cond[0, 1].clone().detach().item() * (82 - 44) + 44,
        inversed_ventricular=cond[0, 2].clone().detach().item(),
        inversed_brain=cond[0, 3].clone().detach().item(),
        writer=writer,
    )

    draw_img(
        target_np,
        title="target",
        step=step_,
        output_folder=OUTPUT_FOLDER,
    )

    draw_img(
        synth_img_corrupted[0, 0].detach().cpu().numpy(),
        title="corrupted",
        step=step_,
        output_folder=OUTPUT_FOLDER,
    )

    writer.flush()
    writer.close()

    torch.save(
        {
            "epoch": step,
            "latent_vectors": latent_vector,
            "optimizer": optimizer_adam.state_dict(),
        },
        OUTPUT_FOLDER / "checkpoint.pth",
    )

    row = [
        hparams.subject_id,
        ssim_,
        psnr_,
        mse_,
        nmse_,
    ]

    with open(
        "/scratch/j/jlevman/jueqi/thesis_experiments/decoder/result_decoder_downsample_2.csv",
        "a",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(row)

    return latent_vector_out


def main(hparams: Namespace) -> None:
    # device = torch.device("cuda" if COMPUTECANADA else "cpu")
    # Don't have enough memory to run on GPU. :(
    device = torch.device("cpu")
    img_tensor = load_target_image(hparams, device)
    writer = SummaryWriter(log_dir=hparams.tensor_board_logger)

    forward = create_corruption_function(hparams=hparams, device=device)
    decoder = load_pre_trained_decoder(
        vae_path=PRETRAINED_MODEL_DECODER_PATH,
        device=device,
    )

    start_time = perf_counter()
    latent_vector_out = project(
        decoder,
        writer=writer,
        hparams=hparams,
        forward=forward,
        target=img_tensor,
        device=device,
        verbose=True,
    )
    print(f"Elapsed: {(perf_counter() - start_time):.1f} s")

    torch.save(
        {"latent_vector_out": latent_vector_out},
        OUTPUT_FOLDER / "latent_vector_out.pth",
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Trainer args", add_help=False)
    add_argument(parser)
    hparams = parser.parse_args()
    # seed_everything(hparams.seed)
    main(hparams)
