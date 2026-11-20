import argparse
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import mlflow.pytorch
import torch
import torch.nn as nn

from models.aekl_no_attention import AutoencoderKL
from models.ddpm_v2_conditioned import DDPM


def _extract_state_dict(ckpt: Any, strip_model_prefix: bool = False) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()
    if isinstance(ckpt, dict):
        sd = (
            ckpt.get("state_dict")
            or ckpt.get("model_state_dict")
            or ckpt.get("ema_state_dict")
            or ckpt
        )
    else:
        raise ValueError(f"Formato checkpoint non supportato: {type(ckpt)}")

    cleaned = {}
    for k, v in sd.items():
        k_clean = k.removeprefix("module.")
        if strip_model_prefix:
            k_clean = k_clean.removeprefix("model.")
        cleaned[k_clean] = v
    return cleaned


def infer_custom_vae_params(
    sd: Dict[str, torch.Tensor],
    resolution: Tuple[int, int, int] = (128, 128, 128),
) -> Tuple[int, Dict[str, Any]]:
    """Deduce embed_dim e hparams per models/aekl_no_attention.py."""
    pq = sd["post_quant_conv.weight"]
    z_channels = int(pq.shape[0])
    embed_dim = int(pq.shape[1])

    w0 = sd["encoder.blocks.0.weight"]
    out_w = [v for k, v in sd.items() if k.startswith("decoder.blocks.") and k.endswith(".weight")][-1]
    in_channels = int(w0.shape[1])
    out_channels = int(out_w.shape[0])

    conv1_keys = sorted(
        [k for k in sd if k.startswith("encoder.blocks.") and k.endswith(".conv1.weight")],
        key=lambda x: int(x.split(".")[2]),
    )
    channels_seq = [int(sd[k].shape[0]) for k in conv1_keys]

    runs = []
    for ch in channels_seq:
        if runs and runs[-1][0] == ch:
            runs[-1][1] += 1
        else:
            runs.append([ch, 1])

    num_res_blocks = runs[0][1]
    num_channels = []
    for ch, count in runs:
        n_sublevels = max(1, count // num_res_blocks)
        num_channels.extend([ch] * n_sublevels)

    n_channels = num_channels[0]
    ch_mult = tuple(c // n_channels for c in num_channels)

    hparams = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "n_channels": n_channels,
        "ch_mult": ch_mult,
        "num_res_blocks": num_res_blocks,
        "z_channels": z_channels,
        "resolution": resolution,
        "attn_resolutions": (),
        "dropout": 0.0,
    }
    return embed_dim, hparams


def infer_ddpm_unet_config(sd: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], str, int]:
    """Deduce unet_config, conditioning_key e timesteps dal checkpoint DDPM LDM."""
    prefix = "model.diffusion_model."
    w_in = sd[f"{prefix}input_blocks.0.0.weight"]
    w_out = sd[f"{prefix}out.2.weight"]

    dims = w_in.ndim - 2
    in_channels = int(w_in.shape[1])
    model_channels = int(w_in.shape[0])
    out_channels = int(w_out.shape[0])
    timesteps = int(sd["betas"].shape[0])

    # Analizza input_blocks per dedurre channel_mult, num_res_blocks e attention_resolutions
    block_ids = sorted(
        {int(m.group(1)) for k in sd if (m := re.match(rf"{re.escape(prefix)}input_blocks\.(\d+)\.", k))}
    )

    levels_channels = []
    levels_resblocks = []
    attention_resolutions = []
    ds = 1
    cur_res_blocks = 0

    for bid in block_ids:
        if bid == 0:
            continue
        is_downsample = any(
            k.startswith(f"{prefix}input_blocks.{bid}.0.op.") for k in sd
        )
        if is_downsample:
            levels_resblocks.append(cur_res_blocks)
            cur_res_blocks = 0
            ds *= 2
        else:
            # È un ResBlock (e opzionalmente uno SpatialTransformer / AttentionBlock)
            w_res = sd[f"{prefix}input_blocks.{bid}.0.out_layers.3.weight"]
            ch = int(w_res.shape[0])
            if cur_res_blocks == 0:
                levels_channels.append(ch)
            cur_res_blocks += 1

            has_attn = any(k.startswith(f"{prefix}input_blocks.{bid}.1.") for k in sd)
            if has_attn and ds not in attention_resolutions:
                attention_resolutions.append(ds)

    levels_resblocks.append(cur_res_blocks)
    channel_mult = tuple(ch // model_channels for ch in levels_channels)
    num_res_blocks = levels_resblocks[0]

    # Verifica cross-attention / conditioning
    context_keys = [k for k in sd if k.startswith(prefix) and k.endswith("attn2.to_k.weight")]
    use_spatial_transformer = len(context_keys) > 0
    context_dim = int(sd[context_keys[0]].shape[1]) if use_spatial_transformer else None

    # Deduce num_heads / num_head_channels
    q_keys = [k for k in sd if k.startswith(prefix) and (k.endswith("attn1.to_q.weight") or k.endswith("qkv.weight"))]
    num_heads = 8
    if q_keys:
        q_dim = int(sd[q_keys[0]].shape[0])
        if q_keys[0].endswith("qkv.weight"):
            q_dim //= 3
        for h in (8, 4, 2, 1):
            if q_dim % h == 0:
                num_heads = h
                break

    unet_config = {
    "params": {
        "image_size": 32,
        "in_channels": 3,          
        "model_channels": 256,
        "out_channels": 3,         
        "num_res_blocks": 3,       
        "attention_resolutions": (),
        "dropout": 0.0,
        "channel_mult": (1, 2),    
        "dims": 3,
        "num_heads": 8,
        "use_spatial_transformer": False,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
    }
}
    if context_dim is not None:
        unet_config["context_dim"] = context_dim

    conditioning_key = "crossattn" if use_spatial_transformer else "None"
    return unet_config, conditioning_key, timesteps


def convert_checkpoint(ckpt_path: str, output_dir: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd = _extract_state_dict(ckpt, strip_model_prefix=False)

    is_vae = any("encoder.blocks." in k for k in raw_sd) and any("decoder.blocks." in k for k in raw_sd)
    is_ddpm = "betas" in raw_sd and any(k.startswith("model.diffusion_model.") for k in raw_sd)

    if is_vae:
        sd = _extract_state_dict(ckpt, strip_model_prefix=True)
        embed_dim, hparams = infer_custom_vae_params(sd)
        print(f"Rilevato AutoencoderKL custom -> embed_dim: {embed_dim}")
        for k, v in hparams.items():
            print(f"  {k}: {v}")
        model = AutoencoderKL(embed_dim=embed_dim, hparams=hparams)
        model.load_state_dict(sd, strict=True)

    elif is_ddpm:
        unet_config, conditioning_key, timesteps = infer_ddpm_unet_config(raw_sd)
        print(f"Rilevato DDPM -> timesteps: {timesteps}, conditioning_key: {conditioning_key}")
        for k, v in unet_config.items():
            print(f"  {k}: {v}")
        if isinstance(unet_config, dict) and "params" not in unet_config:
            unet_config = {"params": unet_config}
        model = DDPM(
            unet_config=unet_config,
            timesteps=timesteps,
            conditioning_key=conditioning_key,
            learn_logvar=True,
        )
        model.load_state_dict(raw_sd, strict=True)

    else:
        raise RuntimeError(f"Modello non riconosciuto. Prime 5 chiavi: {list(raw_sd.keys())[:5]}")

    model.eval()
    print("Pesi caricati con successo (strict=True)!")

    out_path = Path(output_dir)
    if out_path.exists():
        raise FileExistsError(f"La cartella '{out_path}' esiste già.")

    mlflow.pytorch.save_model(pytorch_model=model, path=str(out_path))
    print(f"Pacchetto MLflow salvato correttamente in: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converte checkpoint VAE o DDPM in MLflow.")
    parser.add_argument("--ckpt", required=True, help="Percorso al checkpoint .pth")
    parser.add_argument("--out", required=True, help="Cartella di output per MLflow")
    args = parser.parse_args()

    convert_checkpoint(args.ckpt, args.out)