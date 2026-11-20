import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from models.BRGM.forward_models import ForwardDownsample

def _downsample_tensor(tensor_x: torch.Tensor, downsampler: ForwardDownsample, device: str) -> np.ndarray:
    """Helper interno per gestire la dimensionalità, applicare il downsample e restituire un array numpy."""
    original_ndim = tensor_x.ndim

    # Preparazione della forma 5D per F.interpolate (trilinear): [B, C, D, H, W]
    if original_ndim == 3:
        input_tensor = tensor_x.unsqueeze(0).unsqueeze(0)
    elif original_ndim == 4:
        input_tensor = tensor_x.unsqueeze(0)
    elif original_ndim == 5:
        input_tensor = tensor_x
    else:
        raise ValueError(f"Formato non supportato: {original_ndim}D. Atteso 3D, 4D o 5D.")

    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        tensor_lr = downsampler(input_tensor).cpu()

    # Ripristino delle dimensioni originali
    if original_ndim == 3:
        return tensor_lr.squeeze(0).squeeze(0).numpy()
    elif original_ndim == 4:
        return tensor_lr.squeeze(0).numpy()
    else:
        return tensor_lr.numpy()


def _save_array(
    data: np.ndarray,
    output_path: Path,
    output_format: str,
    header: fits.Header = None,
    downsample_factor: float = None
) -> None:
    """Helper per salvare l'array in formato .npy oppure .fits."""
    output_format = output_format.lower()

    if output_format == "npy":
        target_path = output_path.with_suffix(".npy")
        np.save(target_path, data)

    elif output_format == "fits":
        target_path = output_path.with_suffix(".fits")
        out_header = header.copy() if header is not None else fits.Header()

        if downsample_factor is not None and downsample_factor > 1:
            for idx in [1, 2, 3]:  # Assi spaziali e spettrale scalati da trilinear
                if f"CRPIX{idx}" in out_header:
                    out_header[f"CRPIX{idx}"] = (out_header[f"CRPIX{idx}"] - 0.5) / downsample_factor + 0.5
                if f"CDELT{idx}" in out_header:
                    out_header[f"CDELT{idx}"] *= downsample_factor

        new_hdu = fits.PrimaryHDU(data=data, header=out_header)
        new_hdu.writeto(target_path, overwrite=True)

    else:
        raise ValueError(f"Formato di output non valido: '{output_format}'. Usa 'fits' oppure 'npy'.")


def process_hr_to_lr(
    hr_dir: str | Path,
    lr_dir: str | Path,
    downsampler: ForwardDownsample = None,
    output_format: str = "npy",
    device: str = "cpu",
    hdu_idx: int = 0
) -> None:
    """
    Legge dati da `hr_dir` (.fits, .fits.gz o .npy) e genera i file downsamplati in `lr_dir`.
    """
    hr_path = Path(hr_dir)
    lr_path = Path(lr_dir)
    lr_path.mkdir(parents=True, exist_ok=True)

    # 1. Recupero della lista di file da processare
    if hr_path.is_file():
        files_to_process = [hr_path]
    elif hr_path.is_dir():
        files_to_process = sorted(
            list(hr_path.glob("*.fits")) + 
            list(hr_path.glob("*.fits.gz")) + 
            list(hr_path.glob("*.npy"))
        )
    else:
        raise FileNotFoundError(f"Percorso specificato non trovato: {hr_path}")

    if not files_to_process:
        print(f"Nessun file (.fits, .fits.gz, .npy) trovato in: {hr_path}")
        return

    print(f"Trovati {len(files_to_process)} file in '{hr_path}'. Avvio elaborazione -> {output_format.upper()}...")

    for file_path in files_to_process:
        header = None

        # A. Lettura da FITS
        if file_path.suffix in [".fits", ".gz"] or file_path.name.endswith(".fits.gz"):
            with fits.open(file_path, memmap=True) as hdul:
                header = hdul[hdu_idx].header.copy()
                data_np = np.array(hdul[hdu_idx].data, dtype=np.float32)

        # B. Lettura da NPY
        elif file_path.suffix == ".npy":
            data_np = np.load(file_path).astype(np.float32)

        else:
            continue

        # Pulizia NaN / Inf valida per entrambi i formati
        data_np = np.nan_to_num(data_np, nan=0.0, posinf=0.0, neginf=0.0)

        # C. Applicazione eventuale Downsampling
        factor = None
        if downsampler is not None:
            tensor_x = torch.from_numpy(data_np)
            data_out = _downsample_tensor(tensor_x, downsampler, device)
            factor = downsampler.factor
        else:
            data_out = data_np

        # D. Salvataggio nel formato scelto (FITS o NPY)
        base_filename = file_path.stem if file_path.suffix != ".gz" else Path(file_path.stem).stem
        output_file_path = lr_path / base_filename
        
        _save_array(
            data=data_out,
            output_path=output_file_path,
            output_format=output_format,
            header=header,
            downsample_factor=factor
        )

    print(f"Completato! {len(files_to_process)} file salvati in ({output_format.upper()}): {lr_path}")


# --- Esempi di Utilizzo ---
if __name__ == "__main__":
    DOWNSAMPLE_FACTOR = 4
    downsampler = ForwardDownsample(factor=DOWNSAMPLE_FACTOR)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CASO 1: Input NPY (HR) -> Output NPY (LR) dentro una cartella dedicata in inputs/
    HR_NPY_INPUT = "../../data/inputs/HI_skyldev_128x128x128_stride64/test/npy_patches"
    LR_NPY_OUTPUT = "../../data/inputs/HI_skyldev_128x128x128_stride64/test_LR"

    process_hr_to_lr(
        hr_dir=HR_NPY_INPUT,
        lr_dir=LR_NPY_OUTPUT,
        downsampler=downsampler,
        output_format="npy",
        device=device
    )

    # CASO 2: Input FITS -> Output FITS
    # process_hr_to_lr(
    #     hr_dir="path/to/fits_hr",
    #     lr_dir="path/to/fits_lr",
    #     downsampler=downsampler,
    #     output_format="fits",
    #     device=device
    # )

    # CASO 3: Input FITS -> Output NPY
    # process_hr_to_lr(
    #     hr_dir="path/to/fits_hr",
    #     lr_dir="path/to/npy_lr",
    #     downsampler=downsampler,
    #     output_format="npy",
    #     device=device
    # )