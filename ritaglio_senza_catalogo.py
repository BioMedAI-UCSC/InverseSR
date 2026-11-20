import os
import signal
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# =============================================================================
# GESTIONE INTERRUZIONE FORZATA (Ctrl+C)
# =============================================================================
def hard_stop(sig, frame):
    print("\n[!] Interruzione forzata ricevuta (Ctrl+C). Uscita immediata...")
    os._exit(1)


signal.signal(signal.SIGINT, hard_stop)


# =============================================================================
# GENERAZIONE GIF ANIMATE 3D
# =============================================================================
def create_3d_slice_gif(patch_dir, output_gif_dir, num_samples=3):
    """Crea GIF animate con rotazione 3D del cubo e piani di taglio lungo l'asse Z."""
    os.makedirs(output_gif_dir, exist_ok=True)

    all_files = [f for f in os.listdir(patch_dir) if f.endswith(".npy")]
    if not all_files:
        print(f"❌ Nessun file .npy trovato in: {patch_dir}")
        return

    sample_patches = np.random.choice(
        all_files, size=min(num_samples, len(all_files)), replace=False
    )

    for patch_id in sample_patches:
        file_path = os.path.join(patch_dir, patch_id)

        data = np.load(file_path)
        if data.ndim == 4:
            data = data[0]

        depth_Z, height_Y, width_X = data.shape
        vmax = np.percentile(data, 99.8) if np.max(data) > 0 else 1.0
        vmin = 0.0

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")

        X_grid, Y_grid = np.meshgrid(np.arange(width_X), np.arange(height_Y))

        def update(frame):
            ax.clear()
            z_slice_idx = frame
            slice_2d = data[z_slice_idx, :, :]

            ax.plot_surface(
                X_grid,
                Y_grid,
                np.full_like(X_grid, z_slice_idx),
                rstride=2,
                cstride=2,
                facecolors=plt.cm.inferno(
                    (slice_2d - vmin) / (vmax - vmin + 1e-8)
                ),
                shade=False,
                alpha=0.85,
            )

            ax.set_xlim(0, width_X)
            ax.set_ylim(0, height_Y)
            ax.set_zlim(0, depth_Z)

            ax.set_xlabel("X (Pixel)")
            ax.set_ylabel("Y (Pixel)")
            ax.set_zlabel("Z (Canali / Profondità)")

            angle = (frame / depth_Z) * 360
            ax.view_init(elev=25, azim=angle)
            ax.set_title(
                f"Patch 3D: {patch_id}\nScansione Z = {z_slice_idx}/{depth_Z-1}"
            )

        ani = animation.FuncAnimation(
            fig, update, frames=range(depth_Z), interval=150
        )

        gif_filename = f"3d_cube_{os.path.splitext(patch_id)[0]}.gif"
        save_path = os.path.join(output_gif_dir, gif_filename)

        ani.save(save_path, writer="pillow", fps=6)
        plt.close(fig)
        print(f"✅ GIF 3D salvata: {save_path}")


# =============================================================================
# ESTRAZIONE PATCH FAST & LOW-MEMORY
# =============================================================================
def process_radio_blind_split(
    fits_path,
    output_dir,
    patch_size=(128, 128, 128),
    stride=128,
    splits=(0.7, 0.15, 0.15),  # Ratio per (Train, Val, Test)
    seed=42,
    data_name="",
    output_format="npy",
    skip_empty=False,
    empty_threshold=2e-5,
):
    """Estrae i patch 3D in modalità blind ottimizzata per la memoria RAM e li suddivide in train, val e test set."""
    assert sum(splits) == 1.0, "La somma delle frazioni di split deve fare 1.0"

    print(f"\nApertura FITS in Memory-Map: {fits_path}")
    hdul = fits.open(
        fits_path, memmap=True, mode="readonly", ignore_missing_simple=True
    )
    raw_data_ref = hdul[0].data if hdul[0].data is not None else hdul[1].data
    shape = raw_data_ref.shape

    Z, Y, X = shape[1:] if len(shape) == 4 else shape
    print(f"Dimensioni Cubo: Z={Z}, Y={Y}, X={X}")

    # 1. Scansione rapida e salvataggio solo delle coordinate (Leggerissimo in RAM)
    # 1. Scansione rapida caricando 1 fetta Z intera in RAM alla volta
    valid_coords = []
    patch_count = 0
    skipped_count = 0

    print("Scansione rapida coordinate cubo 3D (Ottimizzata in RAM)...")
    
    # Se il cubo ha 4 dimensioni (1, Z, Y, X), prendiamo il riferimento 3D
    data_3d = raw_data_ref[0] if len(shape) == 4 else raw_data_ref

    for z in tqdm(range(0, Z - patch_size[0] + 1, stride), desc="Z-axis"):
        # CARICAMENTO IN RAM: Carichiamo l'intero blocco Z necessario per il patch (es. 128 fette)
        # Questo fa 1 singola lettura sequenziale da disco invece di centinaia di letture casuali!
        z_chunk = np.array(data_3d[z : z + patch_size[0]], dtype=np.float32)

        for y in range(0, Y - patch_size[1] + 1, stride):
            for x in range(0, X - patch_size[2] + 1, stride):
                
                # Slicing in RAM (istantaneo)
                d_slice = z_chunk[:, y : y + patch_size[1], x : x + patch_size[2]]

                # Check NaN / Vuoto in RAM
                if np.isnan(d_slice).all():
                    skipped_count += 1
                    continue

                if skip_empty:
                    max_val = np.nanmax(np.abs(d_slice))
                    if max_val < (3 * empty_threshold):
                        skipped_count += 1
                        continue

                valid_coords.append(
                    {
                        "base_name": f"patch_{patch_count:06d}",
                        "x": x,
                        "y": y,
                        "z": z,
                    }
                )
                patch_count += 1

    # 2. Calcolo degli Split (Train / Val / Test) sulle sole coordinate
    train_ratio, val_ratio, test_ratio = splits

    train_coords, val_test_coords = train_test_split(
        valid_coords, train_size=train_ratio, random_state=seed, shuffle=True
    )

    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    if len(val_test_coords) <= 1:
        val_coords = val_test_coords
        test_coords = []
    else:
        val_coords, test_coords = train_test_split(
            val_test_coords,
            train_size=relative_val_ratio,
            random_state=seed,
            shuffle=True,
        )

    split_datasets = {
        "train": train_coords,
        "val": val_coords,
        "test": test_coords,
    }

    # 3. Estrazione dei dati e Salvataggio su disco (Un patch alla volta)
    for subset_name, subset_coords in split_datasets.items():
        if not subset_coords:
            continue

        subset_dir = os.path.join(output_dir, subset_name)
        npy_dir = os.path.join(subset_dir, "npy_patches")
        fits_dir = os.path.join(subset_dir, "fits_patches")

        if output_format in ["npy", "both"]:
            os.makedirs(npy_dir, exist_ok=True)
        if output_format in ["fits", "both"]:
            os.makedirs(fits_dir, exist_ok=True)

        records = []
        print(f"\nScrittura patch su disco: [{subset_name.upper()}]...")

        for item in tqdm(subset_coords, desc=f"Salvando {subset_name}"):
            base_name = item["base_name"]
            x, y, z = item["x"], item["y"], item["z"]

            # Estrazione del dato singolo
            if len(shape) == 4:
                d_slice = raw_data_ref[
                    0,
                    z : z + patch_size[0],
                    y : y + patch_size[1],
                    x : x + patch_size[2],
                ]
            else:
                d_slice = raw_data_ref[
                    z : z + patch_size[0],
                    y : y + patch_size[1],
                    x : x + patch_size[2],
                ]

            p_data = np.nan_to_num(
                np.array(d_slice, dtype=np.float32), nan=0.0
            )

            # Salvataggio .npy
            if output_format in ["npy", "both"]:
                file_npy_path = os.path.join(
                    npy_dir, f"{base_name}_{data_name}.npy"
                )
                np.save(file_npy_path, p_data[np.newaxis, ...])

            # Salvataggio .fits
            if output_format in ["fits", "both"]:
                patch_header = fits.Header()
                (
                    patch_header["NAXIS1"],
                    patch_header["NAXIS2"],
                    patch_header["NAXIS3"],
                ) = (patch_size[2], patch_size[1], patch_size[0])
                fits.writeto(
                    os.path.join(fits_dir, f"{base_name}.fits"),
                    p_data,
                    patch_header,
                    overwrite=True,
                )

            records.append(
                {
                    "patch_id": f"{base_name}_{data_name}.npy",
                    "global_x_min": x,
                    "global_x_max": x + patch_size[2],
                    "global_y_min": y,
                    "global_y_max": y + patch_size[1],
                    "global_z_min": z,
                    "global_z_max": z + patch_size[0],
                    "max_intensity": float(np.max(p_data)),
                    "mean_intensity": float(np.mean(p_data)),
                }
            )

        # Salva indice CSV per ogni sotto-insieme
        pd.DataFrame(records).to_csv(
            os.path.join(subset_dir, f"{subset_name}_index.csv"), index=False
        )
        print(
            f" Subset [{subset_name.upper()}]: Salvati {len(subset_coords)} patch in {subset_dir}"
        )

    hdul.close()
    print("\n✅ Elaborazione completata con successo!")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    FILENAME = "sky_ldev"
    DATA_DIR_NAME = "HI_128x128x128_stride64"
    FITS_PATH = "/leonardo_scratch/large/userexternal/gvitanza/sky_ldev_v2.fits"
    BASE_OUT_DIR = f"/leonardo_scratch/large/userexternal/gvitanza/InverseSR/data/inputs/{DATA_DIR_NAME}"

    """ # 1. Generazione e Split dei Patch
    process_radio_blind_split(
        fits_path=FITS_PATH,
        output_dir=BASE_OUT_DIR,
        patch_size=(256, 256, 256),
        stride=128,
        splits=(0.8, 0.1, 0.1),  # 80% Train, 10% Validation, 10% Test
        seed=42,
        data_name=FILENAME,
        output_format="npy",
        skip_empty=True,
    )
 """
    # 2. Generazione GIF animate di verifica dal subset Train
    print("\n--- Generazione GIF di Verifica (Train Set) ---")
    NPY_TRAIN_DIR = os.path.join(BASE_OUT_DIR, "test", "npy_patches")
    PLOT_GIF_DIR = os.path.join(BASE_OUT_DIR, "gifs_verification")

    create_3d_slice_gif(
        patch_dir=NPY_TRAIN_DIR, output_gif_dir=PLOT_GIF_DIR, num_samples=5
    )