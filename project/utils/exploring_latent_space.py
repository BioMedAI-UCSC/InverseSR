import numpy as np
import torch

from utils.const import PRETRAINED_MODEL_FOLDER


def load_latent_vectors_from_ddim() -> np.ndarray:
    latent_vectors = torch.load(
        PRETRAINED_MODEL_FOLDER / "latent_vector_all_samples.pt"
    )
    return latent_vectors.detach().cpu().numpy()
