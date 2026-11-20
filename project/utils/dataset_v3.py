# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from glob import glob
from data.data_analysis import analyze_and_save_stats
import json
from pathlib import Path


def normalize_dynamic(data, norm_mode, stats=None):
    if stats is None:
        stats = {}

    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data).float()
    else:
        data = data.float()

    if data.numel() == 0:
        return data, stats

    if norm_mode in ['minmax_clip', 'clip']:
        if "filtering_bounds" in stats:
            lower_b = stats["filtering_bounds"]["lower_bound"]
            upper_b = stats["filtering_bounds"]["upper_bound"]
        else:
            lower_b = stats.get("lower_bound", float(data.min()))
            upper_b = stats.get("upper_bound", float(data.max()))

        data_clipped = torch.clamp(data, min=lower_b, max=upper_b)

        if norm_mode == 'clip':
            return data_clipped, {'lower_bound': lower_b, 'upper_bound': upper_b}

        elif norm_mode == 'minmax_clip':
            data_norm = (data_clipped - lower_b) / (upper_b - lower_b + 1e-8)
            return data_norm, {'lower_bound': lower_b, 'upper_bound': upper_b}

    elif norm_mode == 'global_robust':
        # 1. Trova i percentili dal dataset (o calcolali sul dato locale)
        # Questo elimina gli outlier estremi prima di scalare tra 0 e 1
        p_low = stats.get('p1', np.percentile(data, 1)) if stats else np.percentile(data, 1)
        p_high = stats.get('p99', np.percentile(data, 99)) if stats else np.percentile(data, 99)
        
        # Clip per rimuovere picchi anomali
        data_clipped = np.clip(data, p_low, p_high)
        
        # Scala l'intervallo compattato tra 0 e 1
        norm_data = (data_clipped - p_low) / (p_high - p_low + 1e-8)
        return norm_data, {"p_low": p_low, "p_high": p_high}

    elif norm_mode == 'log_minmax':
        # Utile se i dati coprono diversi ordini di grandezza (es. segnali di sismica o microtomografia)
        data_pos = np.maximum(data, 0)
        log_data = np.log1p(data_pos)  # log(1 + x) evita log(0)
        
        vmin, vmax = log_data.min(), log_data.max()
        norm_data = (log_data - vmin) / (vmax - vmin + 1e-8)
        return norm_data, {"vmin": vmin, "vmax": vmax}

    elif norm_mode == 'zscore_robust':
        # Standardizzazione basata su Mediana e IQR (più robusta di Media e Std)
        med = np.median(data)
        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        norm_data = (data - med) / (iqr + 1e-8)
        return norm_data, {"median": med, "iqr": iqr}

    elif norm_mode in ['global_1', 'global_2']:
        raw_stats = stats.get("flux_statistics_raw", {}).get("percentiles", stats)
        p_min_key = 'p1' if norm_mode == 'global_1' else 'p2'
        p_max_key = 'p99' if norm_mode == 'global_1' else 'p98'

        p_min = raw_stats.get(p_min_key, stats.get(p_min_key, 0.0))
        p_max = raw_stats.get(p_max_key, stats.get(p_max_key, 1.0))

        if p_max <= p_min:
            p_max = p_min + 1e-8

        data_norm = (data - p_min) / (p_max - p_min + 1e-8)
        return torch.clamp(data_norm, 0.0, 1.0), {'p_min': p_min, 'p_max': p_max}

    elif norm_mode == 'global_arcsinh':
        raw_stats = stats.get("flux_statistics_raw", {}).get("percentiles", stats)
        p_min = raw_stats.get('p5', stats.get('p5', 0.0))
        p_max = raw_stats.get('p95', stats.get('p95', 1.0))
        p_min_arcsinh = raw_stats.get('p5_arcsinh', stats.get('p5_arcsinh', 0.0))
        p_max_arcsinh = raw_stats.get('p95_arcsinh', stats.get('p95_arcsinh', 1.0))

        if p_max_arcsinh <= p_min_arcsinh:
            p_max_arcsinh = p_min_arcsinh + 1e-8

        data_arcsinh = torch.arcsinh(data)
        data_norm = (data_arcsinh - p_min_arcsinh) / (p_max_arcsinh - p_min_arcsinh + 1e-8)

        patch_stats = {
            'p_min': p_min,
            'p_max': p_max,
            'p_min_arcsinh': p_min_arcsinh,
            'p_max_arcsinh': p_max_arcsinh,
        }
        return torch.clamp(data_norm, 0.0, 1.0), patch_stats

    elif norm_mode == 'local':
        p_min = float(data.min())
        p_max = float(torch.quantile(data, 0.99))

        if p_max <= p_min:
            p_max = p_min + 1e-5

        data_norm = (data - p_min) / (p_max - p_min + 1e-8)
        return torch.clamp(data_norm, 0.0, 1.0), {'p_min': p_min, 'p_max': p_max}

    else:
        return data, {}


def create_3d_gaussian_mask(shape, x_c, y_c, z_c, sigma=1.5):
    D, H, W = shape
    z_grid, y_grid, x_grid = np.ogrid[:D, :H, :W]
    dist_sq = (x_grid - x_c)**2 + (y_grid - y_c)**2 + (z_grid - z_c)**2
    gaussian = np.exp(-dist_sq / (2 * (sigma**2)))
    return gaussian.astype(np.float32)


class RadioPatchDataset(Dataset):
    def __init__(
        self,
        data_dir,
        catalogue_path=None,
        in_channels=1,
        norm_mode="log_minmax",
        filter_mode=None,              # None = lascia i voxel intatti (consigliato per VAE/LDM)
        min_signal_voxels=50,          # Scarta le patch vuote (< 50 voxel sopra soglia); 0 = tieni tutte
        only_positive_signal=True,     # True = conta solo voxel > upper_bound (emissioni reali)
        mask_sigma=1.5,
        cached_stats=None,
        auto_compute_stats=True,
        outlier_method="mad_sigma",
        outlier_threshold=3.0,
        stats_batch_size=32,
        stats_num_workers=4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.in_channels = in_channels
        self.norm_mode = norm_mode
        self.filter_mode = filter_mode
        self.min_signal_voxels = min_signal_voxels
        self.only_positive_signal = only_positive_signal
        self.mask_sigma = mask_sigma

        self.stats_json_path = self.data_dir / "dataset_filter_stats.json"
        self.stats_plot_path = self.data_dir / "radio_distribution.png"

        all_paths = sorted(glob(os.path.join(data_dir, "*.npy")))
        self.patch_files = [os.path.basename(p) for p in all_paths]

        if len(self.patch_files) == 0:
            raise FileNotFoundError(f"Nessun file .npy trovato in: {data_dir}")

        # 1. CARICAMENTO CATALOGO
        self.catalog_dict = {}
        self.feature_cols = ["hi_size", "line_flux_integral", "i", "w20"]
        self.stats_catalog = {}

        if catalogue_path and os.path.exists(catalogue_path):
            df_catalog = pd.read_csv(catalogue_path)
            df_catalog.columns = [c.lower().strip() for c in df_catalog.columns]

            if "patch_id" in df_catalog.columns:
                df_catalog = df_catalog.drop_duplicates(subset=["patch_id"], keep="first")
                self.stats_catalog = {
                    col: (df_catalog[col].min(), df_catalog[col].max())
                    for col in self.feature_cols
                    if col in df_catalog.columns
                }
                self.catalog_dict = df_catalog.set_index("patch_id").to_dict(orient="index")

        # 2. CARICAMENTO O CALCOLO AUTOMATICO STATISTICHE
        self.dataset_stats = {}

        if cached_stats is not None:
            if isinstance(cached_stats, dict):
                self.dataset_stats = cached_stats
                print("[INFO] Statistiche caricate direttamente dal dizionario cached_stats.")
            elif isinstance(cached_stats, (str, Path)) and os.path.exists(cached_stats):
                with open(cached_stats, "r") as f:
                    self.dataset_stats = json.load(f)
                print(f"[INFO] Statistiche caricate dal file in cached_stats: {cached_stats}")
        else:
            candidate_files = ["dataset_filter_stats.json", "dataset_outliers_stats.json", "dataset_stat.json"]
            candidate_dirs = [
                str(self.data_dir),
                str(self.data_dir.parent),
                str(self.data_dir.parent.parent),
            ]

            found_json = None
            for d in candidate_dirs:
                for f_name in candidate_files:
                    full_path = os.path.join(d, f_name)
                    if os.path.exists(full_path):
                        found_json = full_path
                        break
                if found_json:
                    break

            if found_json:
                with open(found_json, "r") as f:
                    self.dataset_stats = json.load(f)
                print(f"[INFO] Statistiche caricate da: {found_json}")
            elif auto_compute_stats:
                self.dataset_stats = self._load_or_compute_stats(
                    outlier_method=outlier_method,
                    outlier_threshold=outlier_threshold,
                    batch_size=stats_batch_size,
                    num_workers=stats_num_workers,
                )
            else:
                print("[Warning] Nessun file di statistiche trovato e auto_compute_stats=False. Usato dizionario vuoto.")

        # 3. ESTRAZIONE SOGLIE E SELEZIONE DELLE SOLE PATCH SIGNIFICATIVE
        self.lower_bound, self.upper_bound = self._extract_bounds(self.dataset_stats)

        if self.min_signal_voxels > 0 and self.upper_bound is not None:
            self.patch_files = self._filter_significant_patches()

    def _filter_significant_patches(self):
        """
        Seleziona esclusivamente le patch che contengono almeno `min_signal_voxels`
        significativi (> upper_bound, oppure fuori da [lower_bound, upper_bound]),
        lasciando però i dati interni di ciascuna patch completamente intatti.
        """
        sign_tag = "pos" if self.only_positive_signal else "both"
        cache_name = f"valid_patches_{sign_tag}_min{self.min_signal_voxels}.json"
        cache_path = self.data_dir / cache_name
        total_initial = len(self.patch_files)

        if cache_path.exists():
            with open(cache_path, "r") as f:
                cached_data = json.load(f)
            valid_files = [fn for fn in cached_data.get("valid_files", []) if (self.data_dir / fn).exists()]
            if valid_files:
                print(
                    f"[INFO] Patch-level filter: caricate {len(valid_files)}/{total_initial} patch "
                    f"significative dalla cache ({cache_name})."
                )
                return valid_files

        print(
            f"[INFO] Scansione patch-level su {total_initial} cubi "
            f"(soglia >= {self.min_signal_voxels} voxel > {self.upper_bound:.4e})..."
        )
        valid_files = []

        for filename in self.patch_files:
            path = self.data_dir / filename
            arr = np.load(path, mmap_mode="r")

            if self.only_positive_signal or self.lower_bound is None:
                sig_count = int(np.count_nonzero(arr > self.upper_bound))
            else:
                sig_count = int(np.count_nonzero((arr < self.lower_bound) | (arr > self.upper_bound)))

            if sig_count >= self.min_signal_voxels:
                valid_files.append(filename)

        if len(valid_files) == 0:
            raise ValueError(
                f"Nessuna patch supera `min_signal_voxels={self.min_signal_voxels}`! "
                "Riduci `min_signal_voxels` o `outlier_threshold`."
            )

        with open(cache_path, "w") as f:
            json.dump(
                {
                    "min_signal_voxels": self.min_signal_voxels,
                    "only_positive_signal": self.only_positive_signal,
                    "lower_bound": self.lower_bound,
                    "upper_bound": self.upper_bound,
                    "total_initial": total_initial,
                    "valid_count": len(valid_files),
                    "valid_files": valid_files,
                },
                f,
                indent=4,
            )

        discarded = total_initial - len(valid_files)
        print(
            f"[SUCCESS] Patch significative mantenute: {len(valid_files)}/{total_initial} "
            f"(scartate {discarded} patch vuote). Lista salvata in: {cache_path}"
        )
        return valid_files

    def _extract_bounds(self, stats):
        if not stats:
            return None, None
        bounds = (
            stats.get("gaussian_noise_bounds_discarded")
            or stats.get("filtering_bounds")
            or stats.get("flux_statistics_filtered")
            or stats
        )
        lower_b = bounds.get("noise_lower_bound", bounds.get("lower_bound"))
        upper_b = bounds.get("noise_upper_bound", bounds.get("upper_bound"))
        return lower_b, upper_b

    def _apply_voxel_filter(self, data_tensor):
        """Applicato SOLO se filter_mode è esplicitamente impostato (es. 'clip' o 'keep_outliers_zero')."""
        if not self.filter_mode or self.lower_bound is None or self.upper_bound is None:
            return data_tensor

        if self.filter_mode == "keep_outliers_zero":
            noise_mask = (data_tensor >= self.lower_bound) & (data_tensor <= self.upper_bound)
            data_filtered = data_tensor.clone()
            data_filtered[noise_mask] = 0.0
            return data_filtered

        elif self.filter_mode == "clip":
            return torch.clamp(data_tensor, min=self.lower_bound, max=self.upper_bound)

        return data_tensor

    def _load_or_compute_stats(
        self, outlier_method, outlier_threshold, batch_size, num_workers
    ):
        print(f"\n[INFO] File {self.stats_json_path.name} non trovato in {self.data_dir}.")
        print("[INFO] Avvio calcolo automatico delle statistiche ed estrazione outlier...")

        analyze_and_save_stats(
            dataset=self,
            save_json_path=self.stats_json_path,
            save_plot_path=self.stats_plot_path,
            outlier_method=outlier_method,
            outlier_threshold=outlier_threshold,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        print(f"[SUCCESS] Statistiche salvate in: {self.stats_json_path}\n")

        with open(self.stats_json_path, "r") as f:
            stats_data = json.load(f)

        return stats_data

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        filename = self.patch_files[idx]
        path = os.path.join(self.data_dir, filename)

        data_numpy = np.load(path).astype(np.float32)
        data_numpy = np.nan_to_num(data_numpy, nan=0.0, posinf=0.0, neginf=0.0)

        if data_numpy.ndim == 4 and data_numpy.shape[0] == 1:
            data_numpy = data_numpy.squeeze(0)
        elif data_numpy.ndim == 5:
            data_numpy = data_numpy.squeeze()

        x_raw = torch.from_numpy(data_numpy)

        if self.dataset_stats:
            x_proc = self._apply_voxel_filter(x_raw) if self.filter_mode else x_raw
            x_0, _ = normalize_dynamic(x_proc, self.norm_mode, self.dataset_stats)
        else:
            x_0 = x_raw

        if x_0.ndim == 3:
            x_0 = x_0.unsqueeze(0)

        if self.in_channels == 3 and x_0.shape[0] == 1:
            x_0 = x_0.expand(3, -1, -1, -1)

        D, H, W = x_0.shape[-3], x_0.shape[-2], x_0.shape[-1]

        # Maschera Spaziale 3D
        spatial_mask = torch.zeros((1, D, H, W), dtype=torch.float32)
        if self.catalog_dict and filename in self.catalog_dict:
            row = self.catalog_dict[filename]
            if all(k in row for k in ("rel_x", "rel_y", "rel_z")):
                mask_np = create_3d_gaussian_mask(
                    (D, H, W),
                    x_c=float(row["rel_x"]),
                    y_c=float(row["rel_y"]),
                    z_c=float(row["rel_z"]),
                    sigma=self.mask_sigma,
                )
                spatial_mask = torch.from_numpy(mask_np).unsqueeze(0)

        # Context Vector
        params = []
        if self.catalog_dict and filename in self.catalog_dict:
            row = self.catalog_dict[filename]
            for col in self.feature_cols:
                if col in row and col in self.stats_catalog:
                    c_min, c_max = self.stats_catalog[col]
                    norm_val = (row[col] - c_min) / (c_max - c_min + 1e-8)
                    params.append(np.clip(norm_val, 0, 1))
                else:
                    params.append(0.5)
        else:
            params = [0.5] * len(self.feature_cols)

        context_vector = torch.tensor(params, dtype=torch.float32)

        return {
            "x_0": x_0,
            "spatial_mask": spatial_mask,
            "context": context_vector,
            "stats": self.dataset_stats,
        }