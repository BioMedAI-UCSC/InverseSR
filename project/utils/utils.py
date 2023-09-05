import os
import random
from argparse import Namespace
from random import choice
from pathlib import Path
from typing import Tuple, Dict, List, Any

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from monai.transforms import apply_transform
from tqdm import tqdm

from models.ddim import DDIMSampler
from models.aekl_no_attention import OnlyDecoder
from models.ddpm_v2_conditioned import DDPM
from models.BRGM.forward_models import (
    ForwardDownsample,
    ForwardFillMask,
    ForwardAbstract,
)
from utils.transorms import get_preprocessing
from utils.const import (
    INPUT_FOLDER,
    MASK_FOLDER,
    PRETRAINED_MODEL_VAE_PATH,
    PRETRAINED_MODEL_DDPM_PATH,
    PRETRAINED_MODEL_VGG_PATH,
    PRETRAINED_MODEL_FOLDER,
    OUTPUT_FOLDER,
    LATENT_SHAPE,
)


def transform_img(
    img_path: Path,
    device: torch.device,
) -> Any:
    data = {"image": img_path}
    data = apply_transform(get_preprocessing(device), data)
    return data["image"]


def load_target_image(hparams: Namespace, device: torch.device) -> torch.Tensor:
    if hparams.data_format == "nii":
        img_path = list(INPUT_FOLDER.glob(f"**/*ur_IXI{hparams.subject_id}*T1.nii.gz"))[
            0
        ]
        img_tensor = transform_img(img_path, device=device)
    elif hparams.data_format == "pth":
        img_path = list(INPUT_FOLDER.glob(f"**/*IXI_T1_{hparams.subject_id}.pth"))[0]
        img_tensor = torch.load(img_path, map_location=device)
    elif hparams.data_format == "img":
        img_path = list(INPUT_FOLDER.glob(f"**/*OAS1_0{hparams.subject_id}_MR1_*.img"))[
            0
        ]
        img_tensor = transform_img(img_path, device=device)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def load_vgg_perceptual(
    hparams: Namespace, target: torch.Tensor, device: torch.device
) -> Tuple[Any, torch.Tensor]:
    with open(PRETRAINED_MODEL_VGG_PATH, "rb") as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    target_features = getVggFeatures(hparams, target, vgg16)
    return vgg16, target_features


def create_corruption_function(
    hparams: Namespace, device: torch.device
) -> ForwardAbstract:
    if hparams.corruption == "downsample":
        forward = ForwardDownsample(factor=hparams.downsample_factor)
    elif hparams.corruption == "mask":
        mask = np.load(MASK_FOLDER / f"{hparams.mask_id}.npy")
        forward = ForwardFillMask(mask=mask, device=device)
    else:
        # No corruption
        forward = ForwardFillMask(device=device)
    return forward


def load_latent_vector_stats(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    tmp = torch.load(
        INPUT_FOLDER.parent / "trained_models" / "latent_vector_mean_std.pt",
        # INPUT_FOLDER.parent / "trained_models" / "latent_vector_zero_std.pt",
        map_location=device,
    )
    return tmp["latent_vector_mean"], tmp["latent_vector_std"]


def setup_noise_inputs(
    device: torch.device, hparams: Namespace
) -> Tuple[torch.Tensor, torch.Tensor]:
    # gender = choice([0, 1]) if hparams.update_gender else 1  # F=0, M=1
    # age = random.uniform(44, 82) if hparams.update_age else 55
    # ventricular = random.random() if hparams.update_ventricular else 0.5
    # brain_volume = random.random() if hparams.update_brain else 0.5
    gender = 0.5 if hparams.update_gender else 1  # F=0, M=1
    age = 63 if hparams.update_age else 55
    ventricular = 0.5 if hparams.update_ventricular else 0.5
    brain_volume = 0.5 if hparams.update_brain else 0.5

    age_normalized = (age - 44) / (82 - 44)
    cond = torch.tensor(
        [[gender, age_normalized, ventricular, brain_volume]], device=device
    )  # shape: [1, 4]

    latent_variable = torch.randn(LATENT_SHAPE, device=device)
    return cond, latent_variable


def load_pre_trained_model(
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    decoder = mlflow.pytorch.load_model(
        str(PRETRAINED_MODEL_VAE_PATH),
        map_location=device,
    )
    ddpm = mlflow.pytorch.load_model(
        str(PRETRAINED_MODEL_DDPM_PATH),
        map_location=device,
    )
    decoder.eval()
    ddpm.eval()
    decoder = decoder.to(device)
    ddpm = ddpm.to(device)
    decoder.requires_grad_(False)
    ddpm.requires_grad_(False)
    return ddpm, decoder


def sample_fn(
    diffusion: torch.nn.Module,
    vqvae: Any,
    gender: int,
    age: float,
    ventricular: float,
    brain_volume: float,
    device: torch.device,
):
    print("Sampling brain!")
    print(f"Gender: {gender}")
    print(f"Age: {age}")
    print(f"Ventricular volume: {ventricular}")
    print(f"Brain volume: {brain_volume}")

    age_normalized = (age - 44) / (82 - 44)
    cond = torch.Tensor([[gender, age_normalized, ventricular, brain_volume]])
    cond_crossatten = cond.unsqueeze(1)
    cond_concat = cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    cond_concat = cond_concat.expand(list(cond.shape[0:2]) + list(LATENT_SHAPE[2:]))
    conditioning = {
        "c_concat": [cond_concat.float().to(device)],
        "c_crossattn": [cond_crossatten.float().to(device)],
    }
    latent_variable = torch.randn(LATENT_SHAPE, device=device)

    ddim = DDIMSampler(diffusion)
    num_timesteps = 50  # 50
    latent_vectors, _ = ddim.sample(
        num_timesteps,
        first_img=latent_variable,
        conditioning=conditioning,
        batch_size=1,
        shape=list(LATENT_SHAPE[1:]),
        eta=1.0,
    )

    with torch.no_grad():
        x_hat = vqvae.reconstruct_ldm_outputs(latent_vectors).cpu()

    return latent_variable.numpy(), conditioning, latent_vectors, x_hat.numpy()


def sample_from_ddpm(
    ddpm: DDPM,
    cond: torch.Tensor,
    img: torch.Tensor,
    device: torch.device,
    decoder: OnlyDecoder,
    num_timesteps: int = 1000,
) -> torch.Tensor:
    cond_tmp = cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # shape: [1, 4, 1, 1, 1]
    cond_crossatten = (
        cond_tmp.squeeze(-1).squeeze(-1).squeeze(-1).unsqueeze(1)
    )  # shape: [1, 1, 4]
    cond_concat = torch.tile(cond_tmp, (20, 28, 20))
    conditioning = {
        "c_concat": [cond_concat],
        "c_crossattn": [cond_crossatten],
    }

    for i in tqdm(
        reversed(range(0, num_timesteps)),
        desc="sampling loop time step",
        total=num_timesteps,
    ):
        img = ddpm.p_sample(
            img,
            conditioning,
            torch.full((1,), i, device=device, dtype=torch.long),
            clip_denoised=ddpm.clip_denoised,
        )

    brain_img = decoder.reconstruct_ldm_outputs(img)
    return brain_img


def generating_latent_vector(
    diffusion: torch.nn.Module,
    latent_variable: torch.Tensor,
    conditioning: Dict[str, List[torch.Tensor]],
    batch_size: int,
):
    ddim = DDIMSampler(diffusion)
    num_timesteps = 50
    latent_vectors, _ = ddim.sample(
        num_timesteps,
        first_img=latent_variable,
        conditioning=conditioning,
        batch_size=batch_size,
        shape=list(LATENT_SHAPE[1:]),
        eta=1.0,
        verbose=False,
    )

    return latent_vectors


def sampling_from_ddim(
    ddim: DDIMSampler,
    latent_variable: torch.Tensor,
    decoder: OnlyDecoder,
    cond: torch.Tensor,
    hparams: Namespace,
) -> torch.Tensor:
    cond_tmp = cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # shape: [1, 4, 1, 1, 1]
    cond_crossatten = (
        cond_tmp.squeeze(-1).squeeze(-1).squeeze(-1).unsqueeze(1)
    )  # shape: [1, 1, 4]
    cond_concat = torch.tile(cond_tmp, (20, 28, 20))
    conditioning = {
        "c_concat": [cond_concat],
        "c_crossattn": [cond_crossatten],
    }

    latent_vectors, _ = ddim.sample(
        hparams.ddim_num_timesteps,
        conditioning=conditioning,
        batch_size=1,
        shape=list(LATENT_SHAPE[1:]),
        first_img=latent_variable,
        eta=hparams.ddim_eta,
        verbose=False,
    )
    brain_img = decoder.reconstruct_ldm_outputs(latent_vectors)
    return brain_img


def load_ddpm_latent_vectors(device: torch.device) -> torch.Tensor:
    ddpm_latent_vectors = torch.load(
        INPUT_FOLDER.parent / "trained_models" / "latent_vector_ddpm_samples_100000.pt",
        map_location=device,
    )
    return ddpm_latent_vectors


def inference(
    vqvae: Any,
    latent_vectors: torch.Tensor,
):
    x_hat = vqvae.reconstruct_ldm_outputs(latent_vectors)
    return x_hat


def getVggFeatures(hparams: Namespace, img: torch.Tensor, vgg16: Any) -> torch.Tensor:
    if hparams.corruption == "downsample":
        # For super-resolution task, the downsampled image could be smaller than what is required for
        # the VGG16 architecture, resize it back to the original size
        tmp_img = F.interpolate(img, (160, 224, 160), mode="trilinear")
    else:
        tmp_img = img

    if hparams.perc_dim == "coronal":
        # (1, 1, 160, 224, 160) -> (1, 160, 224, 160) -> (224, 160, 1, 160) -> (224, 1, 160, 160)
        tmp_img = torch.transpose(torch.transpose(tmp_img.squeeze(0), 0, 2), 1, 2)
    elif hparams.perc_dim == "sagittal":
        # (1, 1, 160, 224, 160) -> (1, 160, 224, 160) -> (160, 1, 224, 160) -> (160, 1, 160, 224)
        tmp_img = torch.rot90(torch.transpose(tmp_img.squeeze(0), 0, 1), 1, [2, 3])
    elif hparams.perc_dim == "axial":
        # (1, 1, 160, 224, 160) -> (1, 160, 224, 160) -> (160, 160, 224, 1) -> (160, 1, 224, 160) -> (160, 1, 224, 160)
        tmp_img = torch.rot90(
            torch.transpose(torch.transpose(tmp_img.squeeze(0), 3, 0), 1, 3),
            2,
            [2, 3],
        )
    tmp_img = tmp_img.repeat(1, 3, 1, 1)  # BCDWH

    # Features for synth images.
    features = vgg16(tmp_img, resize_images=False, return_lpips=True)
    return features


@torch.no_grad()
def compute_prior_stats(
    diffusion_path: Path, n_samples: int, batch_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    diffusion = mlflow.pytorch.load_model(
        str(diffusion_path),
        map_location=device,
    )

    latent_vector_list: List[torch.Tensor] = []
    for idx in range(n_samples // batch_size):
        cond_concat_list, cond_crossatten_list, latent_variable_list = [], [], []
        for _ in range(batch_size):
            cond, latent_variable = setup_noise_inputs(device)
            cond_tmp = (
                cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            )  # shape: [1, 4, 1, 1, 1]
            cond_crossatten = (
                cond_tmp.squeeze(-1).squeeze(-1).squeeze(-1).unsqueeze(1)
            )  # shape: [1, 1, 4]
            cond_concat = torch.tile(cond_tmp, (20, 28, 20))
            cond_concat_list.append(cond_concat)
            cond_crossatten_list.append(cond_crossatten)
            latent_variable_list.append(latent_variable)

        cond_concat = torch.cat(cond_concat_list, dim=0)
        cond_crossatten = torch.cat(cond_crossatten_list, dim=0)
        latent_variable = torch.cat(latent_variable_list, dim=0)
        del cond_concat_list, cond_crossatten_list, latent_variable_list
        conditioning = {
            "c_concat": [cond_concat],
            "c_crossattn": [cond_crossatten],
        }
        latent_vector = generating_latent_vector(
            diffusion=diffusion,
            latent_variable=latent_variable,
            conditioning=conditioning,
            batch_size=batch_size,
        )
        latent_vector_list.append(latent_vector.to(torch.device("cpu")))
        print(f"Finish {idx} / {n_samples // batch_size}")

    latent_vector_samples = torch.cat(
        latent_vector_list, dim=0  # shape: [n_samples, 3, 20, 28, 20]
    )
    del latent_vector_list
    latent_vector_mean = latent_vector_samples.mean(dim=0, keepdim=True)
    latent_vector_std = latent_vector_samples.std(dim=0, keepdim=True)
    # Save the latent vectors, mean and std
    torch.save(latent_vector_samples, OUTPUT_FOLDER / "latent_vector_samples.pt")
    torch.save(
        {
            "latent_vector_mean": latent_vector_mean,
            "latent_vector_std": latent_vector_std,
        },
        OUTPUT_FOLDER / "latent_vector_mean_std.pt",
    )
    print("latent_vector_samples shape", latent_vector_samples.shape)
    print("latent_vector_mean shape", latent_vector_mean.shape)
    print("latent_vector_std shape", latent_vector_std.shape)
    del diffusion, latent_vector_samples
    return latent_vector_mean, latent_vector_std


def load_ddpm_model(ddpm_path: Path, device: torch.device) -> torch.nn.Module:
    diffusion = mlflow.pytorch.load_model(
        str(ddpm_path),
        map_location=device,
    )
    diffusion.eval()
    diffusion = diffusion.to(device)
    diffusion.requires_grad_(False)
    return diffusion


def load_pre_trained_decoder(
    vae_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    vqvae = mlflow.pytorch.load_model(
        str(vae_path),
        map_location=device,
    )
    vqvae.eval()
    vqvae = vqvae.to(device)
    vqvae.requires_grad_(False)
    return vqvae


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def draw_img(img: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots()
    ax.imshow(np.rot90(img, 1), cmap="gray")
    ax.axis("off")
    fig.savefig(
        path / f"{title}.png",
        bbox_inches="tight",
        pad_inches=0,
        format="png",
        dpi=300,
    )
    plt.close()
