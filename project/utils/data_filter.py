from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt



def compute_global_percentiles(patches, p_low=5.0, p_high=95.0, sample_ratio=0.2):
    """Calcola i percentili globali p5 e p95 su un campione casuale di patch."""
    n_patches = len(patches)
    if n_patches == 0:
        raise ValueError("Il dataset di patch è vuoto.")

    if sample_ratio < 1.0:
        sample_size = max(1, int(n_patches * sample_ratio))
        indices = np.random.choice(n_patches, size=sample_size, replace=False)
        sample_data = patches[indices]
    else:
        sample_data = patches

    p5 = float(np.percentile(sample_data, p_low))
    p95 = float(np.percentile(sample_data, p_high))
    return p5, p95


def arcsinh_normalize(patch, p5, p95, linear_factor=1.0):
    """Applica la normalizzazione globale arcsinh basata sui percentili p5 e p95."""
    scale_factor = (p95 - p5) if (p95 - p5) != 0 else 1.0

    # Rescaling dei dati grezzi
    data_scaled = (patch - p5) / scale_factor

    # Trasformazione Arcsinh
    norm_patch = np.arcsinh(linear_factor * data_scaled)

    # Mappatura nell'intervallo [0, 1]
    norm_min = np.arcsinh(0.0)
    norm_max = np.arcsinh(linear_factor)

    if norm_max != norm_min:
        norm_patch = (norm_patch - norm_min) / (norm_max - norm_min)

    # Clipping per sicurezza sugli outlier
    return np.clip(norm_patch, 0.0, 1.0)


def filter_black_patches(patches, p5, p95, threshold=0.01, std_threshold=1e-5):
    """Filtra le patch 3D considerate nere o prive di segnale valido."""
    valid_patches = []
    valid_indices = []

    for idx, patch in enumerate(patches):
        norm_patch = arcsinh_normalize(patch, p5, p95)

        max_val = float(np.max(norm_patch))
        std_val = float(np.std(norm_patch))

        # Una patch viene mantenuta se ha sufficiente intensità e variabilità interna
        if max_val > threshold and std_val > std_threshold:
            valid_patches.append(patch)
            valid_indices.append(idx)

    return (
        np.array(valid_patches) if valid_patches else np.empty((0,)),
        np.array(valid_indices, dtype=int),
    )


def plot_patch_inspection(
    patches, file_names, kept_indices, p5, p95, max_plots=16, writer=None
):
    """Genera un plot di verifica mostrando la fetta (slice) centrale di ciascuna patch

    dopo la normalizzazione arcsinh, gestendo automaticamente forme 3D e 4D.
    """
    n_show = min(len(patches), max_plots)
    cols = 4
    rows = int(np.ceil(n_show / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = np.array(axes).reshape(-1)

    kept_set = set(kept_indices)

    for i in range(n_show):
        patch = patches[i]
        norm_patch = arcsinh_normalize(patch, p5, p95)

        # Rimuove dimensioni singole fittizie (es. da (1, 16, 128, 128) a (16, 128, 128))
        norm_patch_sq = np.squeeze(norm_patch)

        # Se la patch è 3D (Depth, Height, Width), estraiamo la slice centrale
        if norm_patch_sq.ndim == 3:
            mid_slice = norm_patch_sq[norm_patch_sq.shape[0] // 2, :, :]
        elif norm_patch_sq.ndim == 2:
            mid_slice = norm_patch_sq
        else:
            raise ValueError(
                f"Forma della patch non supportata per la visualizzazione: {norm_patch.shape}"
            )

        is_kept = i in kept_set
        status = "TENUTA" if is_kept else "SCARTATA"
        color = "green" if is_kept else "red"

        max_v = float(np.max(norm_patch))
        std_v = float(np.std(norm_patch))

        im = axes[i].imshow(mid_slice, cmap="inferno", origin="lower")
        axes[i].set_title(
            f"{file_names[i]}\n[{status}]\nMax: {max_v:.3f} | Std: {std_v:.1e}",
            fontsize=9,
            color=color,
            fontweight="bold",
        )
        axes[i].axis("off")

    # Nascondi subplot inutilizzati
    for j in range(n_show, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        "Controllo Qualità Patch (Fetta Centrale Normalizzata Arcsinh)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    # Salvataggio su file locale
    plot_save_path = "plot_check_patches_train.png"
    plt.savefig(plot_save_path, dpi=150, bbox_inches="tight")
    print(f"\n[OK] Plot di verifica salvato in: {plot_save_path}")

    # Salvataggio su TensorBoard se fornito
    if writer is not None:
        writer.add_figure("Check_Patches/Qualita_Filtraggio", fig, global_step=0)
        print("[OK] Plot inviato con successo a TensorBoard!")

    plt.close(fig)


if __name__ == "__main__":
    # 1. Configurazione percorsi
    DATA_DIR = Path("../../data/inputs/HI_128x128x128_stride64/test/npy_patches")
    SAVE_CLEAN = False  # Imposta a True per salvare le patch valide in una nuova cartella
    OUTPUT_DIR = Path("../../data/inputs/HI_128x128x128_stride64/test_clean")

    # 2. Controllo e caricamento file .npy
    file_paths = sorted(list(DATA_DIR.glob("*.npy")))
    num_patches = len(file_paths)

    if num_patches == 0:
        print(f"[ERRORE] Nessun file .npy trovato nel percorso: {DATA_DIR}")
        exit()

    print(f"Caricate {num_patches} patch dalla cartella: {DATA_DIR}")

    # Carica la matrice globale
    dataset = np.array([np.load(fp) for fp in file_paths])
    file_names = [fp.name for fp in file_paths]

    # 3. Calcolo percentili globali
    p5, p95 = compute_global_percentiles(
        dataset, p_low=5.0, p_high=95.0, sample_ratio=0.2
    )
    print(f"Percentile 5% globale : {p5:.6e}")
    print(f"Percentile 95% globale: {p95:.6e}")

    # 4. Filtraggio patch nere
    clean_dataset, kept_indices = filter_black_patches(
        dataset, p5, p95, threshold=0.01, std_threshold=1e-5
    )

    kept_files = [file_names[i] for i in kept_indices]
    discarded_files = set(file_names) - set(kept_files)

    print("\n--- RISULTATO FILTRAGGIO ---")
    print(f"Patch conservate: {len(kept_files)} / {num_patches}")
    print(f"Patch eliminate : {len(discarded_files)}")

    if discarded_files:
        print(f"File eliminati: {sorted(list(discarded_files))}")

    # 5. Generazione Plot di Controllo (e TensorBoard opzionale)
    
    plot_patch_inspection(
        patches=dataset,
        file_names=file_names,
        kept_indices=kept_indices,
        p5=p5,
        p95=p95,
        max_plots=120  # Numero massimo di patch da mostrare nella griglia
    )
   

    # 6. Salvataggio dei soli file validi
    if SAVE_CLEAN and len(kept_indices) > 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for idx in kept_indices:
            orig_path = file_paths[idx]
            save_path = OUTPUT_DIR / orig_path.name
            np.save(save_path, dataset[idx])
        print(f"[OK] Patch filtrate salvate con successo in: {OUTPUT_DIR}")