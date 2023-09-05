# Code is adpated from: https://huggingface.co/spaces/Warvito/diffusion_brain/blob/main/app.py and
# https://colab.research.google.com/drive/1xJAor6_Ky36gxIk6ICNP--NMBjSlYKil?usp=sharing#scrollTo=4XDeCy-Vj59b
# A lot of thanks to the author of the code

# Reference:
# [1] Pinaya, W. H., et al. (2022). "Brain Imaging Generation with Latent Diffusion Models." arXiv preprint arXiv:2209.07162.
# [2] Marinescu, R., et al. (2020). Bayesian Image Reconstruction using Deep Generative Models.

import math
from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import perf_counter
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import csv
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import apply_transform
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from models.BRGM.forward_models import (
    ForwardDownsample,
    ForwardFillMask,
    ForwardAbstract,
)
from models.ddim import DDIMSampler
from utils.transorms import get_preprocessing
from utils.plot import draw_corrupted_images, draw_images, draw_img
from utils.add_argument import add_argument
from utils.utils import (
    setup_noise_inputs,
    load_target_image,
    load_pre_trained_model,
    create_corruption_function,
    sampling_from_ddim,
    getVggFeatures,
)
from utils.const import (
    INPUT_FOLDER,
    PRETRAINED_MODEL_VGG_PATH,
    OUTPUT_FOLDER,
)


def transform_img(
    img_path: Path,
    device: torch.device,
) -> Any:
    data = {"image": img_path}
    data = apply_transform(get_preprocessing(device), data)
    return data["image"]


def return_true_conditional_variable(subject_id: str) -> Tuple[int, int]:
    csv_path = INPUT_FOLDER / "oasis_cross-sectional.csv"
    df = pd.read_csv(csv_path)
    df = df[df["ID"] == f"{subject_id}_MR1"]
    gender = 0 if df["M/F"].values[0] == "F" else 1
    age = df["Age"].values[0]
    return gender, age.item()


def create_corruption_imgs(
    img_tensor: torch.Tensor, hparams: Namespace, device: torch.device
) -> Tuple[ForwardAbstract, torch.Tensor]:
    if hparams.corruption == "downsample":
        forward = ForwardDownsample(factor=hparams.downsample_factor)
        # mask = skimage.io.imread(maskFile)
        # mask = mask[:, :, 0] == np.min(mask[:, :, 0])  # need to block black color
        # mask = np.reshape(mask, (1, 1, mask.shape[0], mask.shape[1]))

        # Original image for now
        corrupted_img = forward(
            img_tensor
        )  # pass through forward model to generate corrupted image
    elif hparams.corruption == "None":
        forward = ForwardFillMask(device=device)
        corrupted_img = img_tensor
    return forward, corrupted_img


def load_vgg_perceptual(
    hparams: Namespace, target: torch.Tensor, device: torch.device
) -> Tuple[Any, torch.Tensor]:
    with open(PRETRAINED_MODEL_VGG_PATH, "rb") as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    target_features = getVggFeatures(hparams, target, vgg16)
    return vgg16, target_features


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


def project(
    ddim: DDIMSampler,
    decoder: torch.nn.Module,
    forward: ForwardFillMask,
    target: torch.Tensor,
    device: torch.device,
    writer: SummaryWriter,
    hparams: Namespace,
    verbose: bool = False,
):
    cond, latent_variable = setup_noise_inputs(device=device, hparams=hparams)

    update_params = []
    if hparams.update_latent_variables:
        latent_variable.requires_grad = True
        update_params.append(latent_variable)
    if hparams.update_conditioning:
        cond.requires_grad = True
        update_params.append(cond)

    optimizer_adam = torch.optim.Adam(
        update_params,
        betas=(0.9, 0.999),
        lr=hparams.learning_rate,
    )
    latent_variable_out = torch.zeros(
        [hparams.num_steps] + list(latent_variable.shape[1:]),
        dtype=torch.float32,
        device=device,
    )
    cond_out = torch.zeros(
        [hparams.num_steps] + list(cond.shape[1:]),
        dtype=torch.float32,
        device=device,
    )

    mask_cond = create_mask_for_backprop(hparams, device)
    target_img_corrupted = forward(target)
    vgg16, target_features = load_vgg_perceptual(hparams, target_img_corrupted, device)
    total_num_pixels = (
        target_img_corrupted.numel()
        if hparams.corruption != "mask"
        else math.prod(forward.mask.shape) - forward.mask.sum()
    )

    for step in range(hparams.start_steps, hparams.num_steps):

        def closure():
            optimizer_adam.zero_grad()

            synth_img = sampling_from_ddim(
                ddim=ddim,
                decoder=decoder,
                latent_variable=latent_variable,
                cond=cond,
                hparams=hparams,
            )
            synth_img_corrupted = forward(synth_img)  # f(G(w))

            loss = 0
            prior_loss = 0
            pixelwise_loss = (
                synth_img_corrupted - target_img_corrupted
            ).abs().sum() / total_num_pixels
            loss += pixelwise_loss

            synth_features = getVggFeatures(hparams, synth_img_corrupted, vgg16)
            perceptual_loss = (target_features - synth_features).abs().mean()
            loss += hparams.lambda_perc * perceptual_loss

            loss.backward(create_graph=False)
            cond.grad *= mask_cond

            return (
                loss,
                pixelwise_loss,
                perceptual_loss,
                prior_loss,
                synth_img,
                synth_img_corrupted,
            )

        (
            loss,
            pixelwise_loss,
            perceptual_loss,
            prior_loss,
            synth_img,
            synth_img_corrupted,
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
        writer.add_scalar("perceptual_loss", perceptual_loss, global_step=step)
        writer.add_scalar("prior_loss", prior_loss, global_step=step)
        writer.add_scalar("ssim", ssim_, global_step=step)
        writer.add_scalar("psnr", psnr_, global_step=step)
        writer.add_scalar("mse", mse_, global_step=step)
        writer.add_scalar("nmse", nmse_, global_step=step)

        if hparams.update_conditioning:
            if hparams.update_gender:
                writer.add_scalar("inversed_gender", cond[0, 0], global_step=step)
            if hparams.update_age:
                writer.add_scalar("inversed_age", cond[0, 1], global_step=step)
            if hparams.update_ventricular:
                writer.add_scalar("inversed_ventricular", cond[0, 2], global_step=step)
            if hparams.update_brain:
                writer.add_scalar("inversed_brain", cond[0, 3], global_step=step)
        logprint(
            f"step {step + 1:>4d}/{hparams.num_steps}: tloss {float(loss):<5.8f} pix_loss {float(pixelwise_loss):<5.8f} perc_loss {float(perceptual_loss):<1.15f} pior_loss {float(prior_loss):<5.8f}\n"
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

        latent_variable_out[step] = latent_variable.detach()[0]
        cond_out[step] = cond.detach()[0]

    add_hparams_to_tensorboard(
        hparams,
        final_loss=loss.item(),
        final_ssim=ssim_,
        final_psnr=psnr_,
        final_mse=mse_,
        final_nmse=nmse_,
        inversed_ventricular=cond[0, 2].clone().detach().item(),
        inversed_brain=cond[0, 3].clone().detach().item(),
        writer=writer,
    )

    writer.flush()
    writer.close()

    torch.save(
        {
            "epoch": step,
            "latent_variable": latent_variable,
            "cond": cond,
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
        cond[0, 0].clone().detach().item(),
        cond[0, 1].clone().detach().item() * (82 - 44) + 44,
        cond[0, 2].clone().detach().item(),
        cond[0, 3].clone().detach().item(),
    ]

    # Open the file in write mode
    with open(
        # Replace the path with your csv file path
        "/scratch/j/jlevman/jueqi/thesis_experiments/ddim/result_ddim_mask_9.csv",
        "a",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(row)

    return latent_variable_out, cond_out


def main(hparams: Namespace) -> None:
    # device = torch.device("cuda" if COMPUTECANADA else "cpu")
    # Don't have enough memory to run on GPU.
    device = torch.device("cpu")
    img_tensor = load_target_image(hparams, device=device)
    writer = SummaryWriter(log_dir=hparams.tensor_board_logger)

    # Create forward corruption model that masks the image with the given mask
    # Make the mask function work
    forward = create_corruption_function(hparams=hparams, device=device)
    diffusion, decoder = load_pre_trained_model(device=device)
    ddim = DDIMSampler(diffusion)

    # Call projector code
    start_time = perf_counter()
    latent_variable_out, cond_out = project(
        ddim,
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
        {"latent_variable": latent_variable_out, "cond": cond_out},
        OUTPUT_FOLDER / "latent_cond.pth",
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Trainer args", add_help=False)
    add_argument(parser)
    hparams = parser.parse_args()
    # seed_everything(42)
    main(hparams)
