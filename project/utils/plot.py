from typing import List
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def merge_two_images(img1: np.ndarray, img2: np.ndarray, dim: int) -> plt.figure:
    si, sj, sk = img1.shape
    if dim == 1:
        img1_slice = np.rot90(img1[si // 2, ...], -1)
        img2_slice = np.rot90(img2[si // 2, ...], -1)
    elif dim == 2:
        img1_slice = np.rot90(img1[:, sj // 2, :], 1)
        img2_slice = np.rot90(img2[:, sj // 2, :], 1)
    elif dim == 3:
        img1_slice = np.rot90(img1[:, :, sk // 2], 1)
        img2_slice = np.rot90(img2[:, :, sk // 2], 1)
    imgs_list = [img1_slice, img2_slice]
    titles_list = ["Reconstructed Image", "Original Corrupted"]

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 2)
    for idx in range(2):
        ax = plt.subplot(gs[idx])
        ax.imshow(imgs_list[idx], cmap="gray")
        ax.grid(False)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles_list[idx])

    plt.tight_layout()
    return fig


def draw_corrupted_images(
    img1: np.ndarray, img2: np.ndarray, img3: np.ndarray, img4: np.ndarray, ssim_: float
) -> np.ndarray:
    si, sj, sk = img1.shape
    si_, sj_, sk_ = img3.shape
    img1_row1 = np.rot90(img1[:, :, sk // 2], -1)
    img2_row1 = np.rot90(img2[:, :, sk // 2], -1)
    img3_row1 = np.rot90(img3[:, :, sk_ // 2], -1)
    img4_row1 = np.rot90(img4[:, :, sk_ // 2], -1)
    img1_row2 = np.rot90(img1[:, sj // 2, :], -1)
    img2_row2 = np.rot90(img2[:, sj // 2, :], -1)
    img3_row2 = np.rot90(img3[:, sj_ // 2, :], -1)
    img4_row2 = np.rot90(img4[:, sj_ // 2, :], -1)
    img1_row3 = np.rot90(img1[si // 2, :, :], -1)
    img2_row3 = np.rot90(img2[si // 2, :, :], -1)
    img3_row3 = np.rot90(img3[si_ // 2, :, :], -1)
    img4_row3 = np.rot90(img4[si_ // 2, :, :], -1)
    imgs_list = [
        img1_row1,
        img2_row1,
        img3_row1,
        img4_row1,
        img1_row2,
        img2_row2,
        img3_row2,
        img4_row2,
        img1_row3,
        img2_row3,
        img3_row3,
        img4_row3,
    ]
    titles_list = [
        "Reconstructed Image",
        "Original Corrupted",
        "Reconstructed Image (downsampled)",
        "Original Corrupted (downsampled)",
    ]

    fig = plt.figure(figsize=(16, 18))
    nrows, ncols = 3, 4
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    for idx in range(nrows):
        for jdx in range(ncols):
            ax = plt.subplot(gs[idx * ncols + jdx])
            ax.imshow(imgs_list[idx * ncols + jdx], cmap="gray")
            ax.grid(False)
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            if idx == 0:
                ax.set_title(titles_list[idx * ncols + jdx])

    plt.tight_layout()
    fig.suptitle(f"SSIM: {ssim_:.4f}", x=0.48, y=0.99, fontsize=12)
    return fig


def draw_images_for_variational_inference(
    corrupted: np.ndarray, target: np.ndarray, synth_imgs: np.ndarray, ssim_: float
) -> np.ndarray:
    _, _, si, sj, sk = target.shape
    _, _, si_, sj_, sk_ = corrupted.shape
    print(f"synth_imgs.shape: {synth_imgs.shape}")
    img1_row1 = np.rot90(corrupted[0, 0, :, :, sk_ // 2], -1)
    img2_row1 = np.rot90(target[0, 0, :, :, sk // 2], -1)
    img3_row1 = np.rot90(synth_imgs[0, 0, :, :, sk // 2], -1)
    img4_row1 = np.rot90(synth_imgs[1, 0, :, :, sk // 2], -1)
    img5_row1 = np.rot90(synth_imgs[2, 0, :, :, sk // 2], -1)
    img6_row1 = np.rot90(synth_imgs[3, 0, :, :, sk // 2], -1)
    img1_row2 = np.rot90(corrupted[0, 0, :, sj_ // 2, :], -1)
    img2_row2 = np.rot90(target[0, 0, :, sj // 2, :], -1)
    img3_row2 = np.rot90(synth_imgs[0, 0, :, sj // 2, :], -1)
    img4_row2 = np.rot90(synth_imgs[1, 0, :, sj // 2, :], -1)
    img5_row2 = np.rot90(synth_imgs[2, 0, :, sj // 2, :], -1)
    img6_row2 = np.rot90(synth_imgs[3, 0, :, sj // 2, :], -1)
    img1_row3 = np.rot90(corrupted[0, 0, si_ // 2, :, :], -1)
    img2_row3 = np.rot90(target[0, 0, si // 2, :, :], -1)
    img3_row3 = np.rot90(synth_imgs[0, 0, si // 2, :, :], -1)
    img4_row3 = np.rot90(synth_imgs[1, 0, si // 2, :, :], -1)
    img5_row3 = np.rot90(synth_imgs[2, 0, si // 2, :, :], -1)
    img6_row3 = np.rot90(synth_imgs[3, 0, si // 2, :, :], -1)
    imgs_list = [
        img1_row1,
        img2_row1,
        img3_row1,
        img4_row1,
        img5_row1,
        img6_row1,
        img1_row2,
        img2_row2,
        img3_row2,
        img4_row2,
        img5_row2,
        img6_row2,
        img1_row3,
        img2_row3,
        img3_row3,
        img4_row3,
        img5_row3,
        img6_row3,
    ]
    titles_list = [
        "Corrupted Image",
        "Target Image",
        "Est. Mean",
        "Sample 1",
        "Sample 2",
        "Sample 3",
    ]

    fig = plt.figure(figsize=(24, 18))
    nrows, ncols = 3, 6
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    for idx in range(nrows):
        for jdx in range(ncols):
            ax = plt.subplot(gs[idx * ncols + jdx])
            ax.imshow(imgs_list[idx * ncols + jdx], cmap="gray")
            ax.grid(False)
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            if idx == 0:
                ax.set_title(titles_list[idx * ncols + jdx])

    plt.tight_layout()
    fig.suptitle(f"SSIM: {ssim_:.4f}", x=0.48, y=0.99, fontsize=12)
    return fig


def draw_images(
    img1: np.ndarray,
    img2: np.ndarray,
    ssim_: float,
    titles_list: List[str] = [
        "Reconstructed Image",
        "Original Corrupted",
    ],
) -> np.ndarray:
    si, sj, sk = img1.shape
    img1_row1 = np.rot90(img1[:, :, sk // 2], -1)
    img2_row1 = np.rot90(img2[:, :, sk // 2], -1)
    img1_row2 = np.rot90(img1[:, sj // 2, :], -1)
    img2_row2 = np.rot90(img2[:, sj // 2, :], -1)
    img1_row3 = np.rot90(img1[si // 2, :, :], -1)
    img2_row3 = np.rot90(img2[si // 2, :, :], -1)
    imgs_list = [
        img1_row1,
        img2_row1,
        img1_row2,
        img2_row2,
        img1_row3,
        img2_row3,
    ]

    fig = plt.figure(figsize=(8, 18))
    nrows, ncols = 3, 2
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    for idx in range(nrows):
        for jdx in range(ncols):
            ax = plt.subplot(gs[idx * ncols + jdx])
            ax.imshow(imgs_list[idx * ncols + jdx], cmap="gray")
            ax.grid(False)
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            if idx == 0:
                ax.set_title(titles_list[idx * ncols + jdx])

    plt.tight_layout()
    fig.suptitle(f"SSIM: {ssim_:.4f}", x=0.48, y=0.99, fontsize=12)
    return fig


def draw_img(img: np.ndarray, title: str, step: str, output_folder: Path) -> None:
    fig, ax = plt.subplots()
    si, sj, sk = img.shape
    img_slice = np.rot90(img[:, sj // 2, :], -1)
    ax.imshow(img_slice, cmap="gray")
    ax.axis("off")
    fig.savefig(
        output_folder / f"{step}_{title}.png",
        bbox_inches="tight",
        pad_inches=0,
        format="png",
        dpi=300,
    )
    # close
    plt.close(fig)


def draw_line_plot(
    img_path: Path,
) -> None:
    img = cv2.imread(str(img_path))
    ix, iy, _ = img.shape
    img_with_line = cv2.line(img, (iy // 2, 0), (iy // 2, ix), (0, 0, 255), 2)
    img_with_line = cv2.line(img, (0, ix // 2), (iy, ix // 2), (0, 0, 255), 2)
    cv2.imwrite(str(img_path), img_with_line)


def draw_img_in_three_dim(img: np.ndarray, title: str, output_folder: Path) -> None:
    fig, ax = plt.subplots()
    si, sj, sk = img.shape
    dim_slices = ["axial", "sagittal", "coronal"]

    img_slice = np.rot90(img[:, :, sk // 2], 1)
    ax.imshow(img_slice, cmap="gray")
    ax.axis("off")
    fig.savefig(
        output_folder / f"{title}_{dim_slices[0]}.png",
        bbox_inches="tight",
        pad_inches=0,
        format="png",
        dpi=300,
    )
    # draw_line_plot(output_folder / f"{title}_{dim_slices[0]}.png")

    img_slice = np.rot90(img[:, sj // 2, :], 1)
    ax.imshow(img_slice, cmap="gray")
    ax.axis("off")
    fig.savefig(
        output_folder / f"{title}_{dim_slices[1]}.png",
        bbox_inches="tight",
        pad_inches=0,
        format="png",
        dpi=300,
    )
    # draw_line_plot(output_folder / f"{title}_{dim_slices[1]}.png")

    img_slice = np.rot90(img[si // 2, :, :], 1)
    ax.imshow(img_slice, cmap="gray")
    ax.axis("off")
    fig.savefig(
        output_folder / f"{title}_{dim_slices[2]}.png",
        bbox_inches="tight",
        pad_inches=0,
        format="png",
        dpi=300,
    )
    # draw_line_plot(output_folder / f"{title}_{dim_slices[2]}.png")
    plt.close(fig)


def draw_three_imgs_one_row(
    img1: np.ndarray,
    img2: np.ndarray,
    img3: np.ndarray,
    titles_list: List[str] = [
        "GT",
        "Input_nearest",
        "SynthSR",
    ],
) -> np.ndarray:
    si, sj, sk = img1.shape
    img1_row1 = np.rot90(img1[:, sj // 2, :], -1)
    img2_row2 = np.rot90(img2[:, sj // 2, :], -1)
    img3_row3 = np.rot90(img3[:, sj // 2, :], -1)
    imgs_list = [
        img1_row1,
        img2_row2,
        img3_row3,
    ]

    fig = plt.figure(figsize=(18, 8))
    nrows, ncols = 1, 3
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    for idx in range(nrows):
        for jdx in range(ncols):
            ax = plt.subplot(gs[idx * ncols + jdx])
            ax.imshow(imgs_list[idx * ncols + jdx], cmap="gray")
            ax.grid(False)
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(titles_list[idx * ncols + jdx])

    plt.tight_layout()
    return fig
