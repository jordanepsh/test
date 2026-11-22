"""Diffusers LoRA loading and scale-aware merging for MiniMaxH3ImageTransformer2DModel."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

_LORA_A_SUFFIXES = (".lora_A.weight", ".lora_A.default.weight", ".lora.down.weight", ".lora_down.weight")
_LORA_B_SUFFIXES = (".lora_B.weight", ".lora_B.default.weight", ".lora.up.weight", ".lora_up.weight")
_CHUNK_ROWS = 4096


@dataclass
class LoRAEntry:
    """One linear layer's LoRA factors kept on CPU plus the multiplier currently baked into its weight."""

    base: str
    module: nn.Linear
    a_cpu: torch.Tensor
    b_cpu: torch.Tensor
    rank_scale: float
    applied_scale: float = 0.0


def _split_lora_key(key: str) -> tuple[str, str]:
    for suffix in _LORA_A_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)].removeprefix("transformer."), "A"
    for suffix in _LORA_B_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)].removeprefix("transformer."), "B"
    if key.endswith(".alpha"):
        return key[: -len(".alpha")].removeprefix("transformer."), "alpha"
    raise ValueError(f"Unrecognized LoRA key: {key!r}")


def _module_lora_multiplier(
    base: str,
    attn_scale: float,
    mlp_scale: float,
    early_blocks: int,
    early_scale: float,
) -> float:
    """Compute per-module scale multiplier separating attention, FFN/MLP, and early transformer blocks."""
    mult = 1.0
    if ".attn." in base:
        mult *= attn_scale
    elif ".ff." in base:
        mult *= mlp_scale
    if early_blocks > 0 and early_scale != 1.0 and base.startswith("transformer_blocks."):
        parts = base.split(".", 2)
        if len(parts) >= 2 and parts[1].isdigit() and int(parts[1]) < early_blocks:
            mult *= early_scale
    return mult


def load_lora_entries_from_state_dict(transformer: nn.Module, lora_sd: dict[str, torch.Tensor]) -> list[LoRAEntry]:
    """Validate Diffusers LoRA tensors against `transformer` Linear modules and stage CPU `(A, B)` factors."""
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    alphas: dict[str, float] = {}
    for key, tensor in lora_sd.items():
        base, kind = _split_lora_key(key)
        if kind == "alpha":
            alphas[base] = float(tensor.item())
        else:
            pairs.setdefault(base, {})[kind] = tensor
    if not pairs:
        raise ValueError("LoRA state dict contains no lora_A/lora_B weight pairs.")

    entries: list[LoRAEntry] = []
    for base, entry in pairs.items():
        if "A" not in entry or "B" not in entry:
            raise ValueError(f"Incomplete LoRA pair for {base!r}: keys={tuple(entry.keys())}")
        module = transformer.get_submodule(base)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"Target module {base!r} must be nn.Linear, got {type(module).__name__}")
        a, b = entry["A"], entry["B"]
        w = module.weight.detach()
        if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
            raise ValueError(f"Invalid LoRA shapes for {base!r}: B={tuple(b.shape)}, A={tuple(a.shape)}")
        if (b.shape[0], a.shape[1]) != tuple(w.shape):
            raise ValueError(f"Shape mismatch for {base!r}: LoRA {(b.shape[0], a.shape[1])} vs weight {tuple(w.shape)}")
        rank = a.shape[0]
        rank_scale = alphas.get(base, float(rank)) / float(rank)
        entries.append(
            LoRAEntry(
                base=base,
                module=module,
                a_cpu=a.detach().cpu().contiguous(),
                b_cpu=b.detach().cpu().contiguous(),
                rank_scale=rank_scale,
            )
        )
    return entries


def load_lora_entries(transformer: nn.Module, lora_path: str) -> list[LoRAEntry]:
    """Load a Diffusers `.safetensors` LoRA file into `LoRAEntry` records (~158 MiB CPU overhead)."""
    from safetensors.torch import load_file

    lora_sd = load_file(lora_path, device="cpu")
    try:
        return load_lora_entries_from_state_dict(transformer, lora_sd)
    finally:
        lora_sd.clear()


@torch.no_grad()
def apply_lora_multipliers(
    entries: list[LoRAEntry],
    scale: float = 1.0,
    attn_scale: float = 1.0,
    mlp_scale: float = 1.0,
    early_blocks: int = 10,
    early_scale: float = 1.0,
) -> int:
    """Update merged transformer weights in row-chunked FP32 to match the target LoRA scales."""
    updated = 0
    for entry in entries:
        layer_mult = _module_lora_multiplier(entry.base, attn_scale, mlp_scale, early_blocks, early_scale)
        target = float(scale) * layer_mult * entry.rank_scale
        delta = target - entry.applied_scale
        if abs(delta) <= 1e-7:
            continue
        w = entry.module.weight.detach()
        b_f32 = entry.b_cpu.to(device=w.device, dtype=torch.float32)
        a_f32 = entry.a_cpu.to(device=w.device, dtype=torch.float32)
        for row_start in range(0, w.shape[0], _CHUNK_ROWS):
            row_end = min(row_start + _CHUNK_ROWS, w.shape[0])
            chunk = w[row_start:row_end].to(torch.float32)
            chunk.addmm_(b_f32[row_start:row_end], a_f32, alpha=delta)
            w[row_start:row_end].copy_(chunk.to(w.dtype))
        entry.applied_scale = target
        updated += 1
    return updated
