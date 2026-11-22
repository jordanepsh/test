"""Kernel-level and memory optimizations for MiniMaxAI/MiniMax-H3-Image on ZeroGPU.

Adapted for MiniMaxH3ImageTransformer2DModel and AutoencoderKLMiniMaxH3Image:
- Precomputed per-step AdaLN modulation tables (drops ~13B AdaLN projection weights).
- Rowwise FP8 W8A8 linear layers (torch._scaled_mm / cuBLASLt) across all 50 transformer blocks.
- Fused QKV projection and 5 Triton kernels around attention and SwiGLU FFN.
- Batched VAE spatial tile decoding on AutoencoderKLMiniMaxH3Image._decode.
"""

from __future__ import annotations

import gc
import inspect
import time
import types

import torch
import triton
import triton.language as tl
from torch import nn

from lora_merge import apply_lora_multipliers, load_lora_entries

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0

_W8A16_CONFIGS = [
    triton.Config({"BLOCK_N": n, "BLOCK_K": k}, num_warps=w, num_stages=4)
    for n in (16, 32, 64)
    for k in (128, 256)
    for w in (4, 8)
]
_AUTOTUNE_CACHE = {"cache_results": True} if "cache_results" in inspect.signature(triton.autotune).parameters else {}


def _free_cache() -> None:
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - ZeroGPU emulates CUDA outside GPU calls
        pass


# AdaLN tables ---------------------------------------------------------------------------------------------------------


class AdaLNTable(nn.Module):
    """Returns precomputed AdaLN modulation tensors for the active step's timestep signature."""

    def __init__(self, tables: list[torch.Tensor], hidden_size: int, lookup: dict):
        super().__init__()
        for index, table in enumerate(tables):
            self.register_buffer(f"table_{index}", table, persistent=False)
        self.hidden_size = hidden_size
        self.lookup = lookup

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        table = getattr(self, f"table_{self.lookup['index']}")
        return table.view(-1, 6 * self.hidden_size).chunk(6, dim=-1)


def build_adaln_calls(pipe) -> list[torch.Tensor]:
    """Derive the exact discrete timestep tensors used by MiniMaxH3ImageSetTimestepsStep."""
    timesteps = torch.tensor([1.0 - s for s in pipe.config.sample_sigmas[:-1]], dtype=torch.float32)
    sigmas = torch.cat([1.0 - timesteps, torch.zeros(1)])
    pipe.scheduler.set_timesteps(sigmas=sigmas, device="cpu")
    ref_t = torch.tensor(float(pipe.reference_noise_aug), dtype=torch.float32)
    calls: list[torch.Tensor] = []
    for t in pipe.scheduler.timesteps:
        t_scalar = torch.tensor(float(t), dtype=torch.float32)
        calls.append(t_scalar.unsqueeze(0))
        calls.append(torch.unique(torch.stack([t_scalar, ref_t]), sorted=True))
    return calls


@torch.no_grad()
def install_adaln_tables(transformer: nn.Module, calls: list[torch.Tensor]) -> int:
    """Precompute all transformer block AdaLN projections and drop the linear weights."""
    device = next(transformer.parameters()).device
    time_embed_dtype = transformer.time_embedder.linear_1.weight.dtype
    keys: dict[tuple[float, ...], int] = {}
    tembs: list[torch.Tensor] = []
    for timestep in calls:
        key = tuple(timestep.tolist())
        if key not in keys:
            keys[key] = len(keys)
            temb = transformer.time_proj(timestep.to(device))
            tembs.append(transformer.time_embedder(temb.to(time_embed_dtype)))

    lookup: dict[str, object] = {"keys": keys, "index": None}
    freed = 0
    for block in transformer.transformer_blocks:
        proj = block.adaln_proj
        dtype = proj.linear.weight.dtype
        tables = [proj.linear(nn.functional.silu(temb).to(dtype)).contiguous() for temb in tembs]
        freed += sum(p.numel() for p in proj.parameters())
        block.adaln_proj = AdaLNTable(tables, proj.hidden_size, lookup)
        del proj

    def set_index(module, args, kwargs):
        key = tuple(kwargs["timestep"].tolist())
        idx = keys.get(key)
        if idx is None:
            for known, cand_idx in keys.items():
                if len(known) == len(key) and all(abs(a - b) <= 1e-5 for a, b in zip(known, key)):
                    idx = cand_idx
                    break
        if idx is None:
            raise ValueError(f"No AdaLN table for timesteps {key}; known: {list(keys)}")
        lookup["index"] = idx

    transformer.register_forward_pre_hook(set_index, with_kwargs=True)
    _free_cache()
    return freed


# FP8 W8A8 quantization ------------------------------------------------------------------------------------------------


@triton.jit
def _rowwise_fp8_kernel(x_ptr, q_ptr, s_ptr, K, stride_x, stride_q, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12)
    tl.store(q_ptr + row * stride_q + offs, (x * (448.0 / amax)).to(tl.float8e4nv), mask=mask)
    tl.store(s_ptr + row, amax / 448.0)


def rowwise_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize `(M, K)` BF16 rows into `(M, K)` FP8 e4m3fn + `(M, 1)` FP32 row scales."""
    if x.stride(-1) != 1:
        x = x.contiguous()
    rows, cols = x.shape
    q = torch.empty((rows, cols), device=x.device, dtype=FP8)
    scale = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(cols)
    _rowwise_fp8_kernel[(rows,)](
        x,
        q,
        scale,
        cols,
        x.stride(0),
        q.stride(0),
        BLOCK=block,
        num_warps=8 if block <= 8192 else 16,
    )
    return q, scale


@triton.autotune(configs=_W8A16_CONFIGS, key=["N", "K"], **_AUTOTUNE_CACHE)
@triton.jit
def _w8a16_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_x,
    stride_w,
    stride_y,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, 16)
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_w + offs_k[None, :],
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.bfloat16)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_x + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w))
    acc *= tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)[None, :]
    tl.store(
        y_ptr + offs_m[:, None] * stride_y + offs_n[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def w8a16_gemv(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Fallback FP8 weight-only GEMV for short sequences (`rows <= 16`)."""
    rows, cols = x.shape
    out_features = weight.shape[0]
    x = x.contiguous()
    y = torch.empty((rows, out_features), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(out_features, meta["BLOCK_N"]),)
    _w8a16_kernel[grid](
        x,
        weight,
        scale,
        bias if bias is not None else scale,
        y,
        rows,
        out_features,
        cols,
        x.stride(0),
        weight.stride(0),
        y.stride(0),
        HAS_BIAS=bias is not None,
    )
    return y


class FP8Linear(nn.Module):
    """Per-output-channel FP8 weight + dynamic rowwise FP8 activation linear layer."""

    def __init__(self, linear: nn.Linear, fast_accum: bool = False):
        super().__init__()
        if linear.in_features % 16 != 0 or linear.out_features % 16 != 0:
            raise ValueError(
                f"FP8Linear requires dimensions divisible by 16, got {(linear.out_features, linear.in_features)}."
            )
        weight = linear.weight.detach()
        scale = weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
        self.register_buffer("weight", (weight.float() / scale).to(FP8).contiguous())
        self.register_buffer("weight_scale", scale.t().contiguous())
        self.out_dtype = weight.dtype
        self.bias = linear.bias.detach().to(self.out_dtype) if linear.bias is not None else None
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.fast_accum = fast_accum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        rows = x.reshape(-1, shape[-1])
        if rows.shape[0] <= 16:
            return w8a16_gemv(rows.to(self.out_dtype), self.weight, self.weight_scale.view(-1), self.bias).view(
                *shape[:-1], self.out_features
            )
        q, scale = rowwise_fp8(rows)
        out = torch._scaled_mm(
            q,
            self.weight.t(),
            scale_a=scale,
            scale_b=self.weight_scale,
            bias=self.bias,
            out_dtype=self.out_dtype,
            use_fast_accum=self.fast_accum,
        )
        return out.view(*shape[:-1], self.out_features)


@torch.no_grad()
def quantize_fp8(module: nn.Module, fast_accum: bool = False, skip: tuple[str, ...] = ()) -> int:
    """Replace `nn.Linear` submodules under `module` (except `skip`) with `FP8Linear`."""
    count = 0
    for name, child in list(module.named_modules()):
        for child_name, leaf in list(child.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(leaf, nn.Linear) and not any(s in full for s in skip):
                setattr(child, child_name, FP8Linear(leaf, fast_accum=fast_accum))
                del leaf
                count += 1
    _free_cache()
    return count


# Fused DiT block kernels ----------------------------------------------------------------------------------------------


@triton.jit
def _rms_mod_fp8_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    shift_ptr,
    idx_ptr,
    q_ptr,
    s_ptr,
    n_cols,
    stride_x,
    stride_tab,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tab = tl.load(idx_ptr + row).to(tl.int64) * stride_tab + cols
    sc = tl.load(scale_ptr + tab, mask=mask, other=0.0).to(tl.float32)
    sh = tl.load(shift_ptr + tab, mask=mask, other=0.0).to(tl.float32)
    y = x * tl.math.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps) * w * (1.0 + sc) + sh
    amax = tl.maximum(tl.max(tl.where(mask, tl.abs(y), 0.0), axis=0), 1e-12)
    tl.store(q_ptr + row * n_cols + cols, (y * (448.0 / amax)).to(tl.float8e4nv), mask=mask)
    tl.store(s_ptr + row, amax / 448.0)


@triton.jit
def _res_gate_rms_mod_fp8_kernel(
    h_ptr,
    b_ptr,
    gate_ptr,
    w_ptr,
    scale_ptr,
    shift_ptr,
    idx_ptr,
    out_ptr,
    q_ptr,
    s_ptr,
    n_cols,
    stride_h,
    stride_b,
    stride_tab,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    tab = tl.load(idx_ptr + row).to(tl.int64) * stride_tab + cols
    h = tl.load(h_ptr + row * stride_h + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * stride_b + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gate_ptr + tab, mask=mask, other=0.0).to(tl.float32)
    h = (h + g * b).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * n_cols + cols, h, mask=mask)
    h = h.to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    sc = tl.load(scale_ptr + tab, mask=mask, other=0.0).to(tl.float32)
    sh = tl.load(shift_ptr + tab, mask=mask, other=0.0).to(tl.float32)
    y = h * tl.math.rsqrt(tl.sum(h * h, axis=0) / n_cols + eps) * w * (1.0 + sc) + sh
    amax = tl.maximum(tl.max(tl.where(mask, tl.abs(y), 0.0), axis=0), 1e-12)
    tl.store(q_ptr + row * n_cols + cols, (y * (448.0 / amax)).to(tl.float8e4nv), mask=mask)
    tl.store(s_ptr + row, amax / 448.0)


@triton.jit
def _res_gate_kernel(
    h_ptr,
    b_ptr,
    gate_ptr,
    idx_ptr,
    out_ptr,
    n_cols,
    stride_h,
    stride_b,
    stride_tab,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    tab = tl.load(idx_ptr + row).to(tl.int64) * stride_tab + cols
    h = tl.load(h_ptr + row * stride_h + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * stride_b + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gate_ptr + tab, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * n_cols + cols, (h + g * b).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_fp8_kernel(x_ptr, q_ptr, s_ptr, n_cols, stride_x, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    v = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(x_ptr + row * stride_x + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    y = v * g * tl.sigmoid(g)
    amax = tl.maximum(tl.max(tl.where(mask, tl.abs(y), 0.0), axis=0), 1e-12)
    tl.store(q_ptr + row * n_cols + cols, (y * (448.0 / amax)).to(tl.float8e4nv), mask=mask)
    tl.store(s_ptr + row, amax / 448.0)


@triton.jit
def _qk_norm_rope_kernel(
    qkv_ptr,
    wq_ptr,
    wk_ptr,
    cos_ptr,
    sin_ptr,
    q_ptr,
    k_ptr,
    stride_qkv,
    stride_rope,
    heads,
    eps,
    D: tl.constexpr,
    ROT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    which = tl.program_id(1)
    is_k = which >= heads
    head = which - heads * is_k.to(tl.int32)
    c = tl.arange(0, D)
    half = ROT // 2
    partner = tl.where(c < half, c + half, tl.where(c < ROT, c - half, c))
    base = qkv_ptr + row * stride_qkv + (is_k.to(tl.int64) * heads + head) * D
    x = tl.load(base + c).to(tl.float32)
    xp = tl.load(base + partner).to(tl.float32)
    w = tl.where(is_k, tl.load(wk_ptr + c), tl.load(wq_ptr + c)).to(tl.float32)
    wp = tl.where(is_k, tl.load(wk_ptr + partner), tl.load(wq_ptr + partner)).to(tl.float32)
    r = tl.math.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    xn, xpn = x * r * w, xp * r * wp
    rot = c < ROT
    cos = tl.load(cos_ptr + row * stride_rope + c, mask=rot, other=1.0)
    sin = tl.load(sin_ptr + row * stride_rope + c, mask=rot, other=0.0)
    y = (xn * cos + tl.where(c < half, -xpn, xpn) * sin).to(q_ptr.dtype.element_ty)
    offset = (row * heads + head) * D + c
    tl.store(q_ptr + offset, y, mask=(c < D) & (which < heads))
    tl.store(k_ptr + offset, y, mask=(c < D) & (which >= heads))


def _row_launch(kernel, rows: int, cols: int, *args, **kwargs) -> None:
    block = triton.next_power_of_2(cols)
    kernel[(rows,)](*args, BLOCK=block, num_warps=16 if block >= 16384 else 8, **kwargs)


def _fp8_gemm(q: torch.Tensor, s: torch.Tensor, linear: FP8Linear) -> torch.Tensor:
    return torch._scaled_mm(
        q,
        linear.weight.t(),
        scale_a=s,
        scale_b=linear.weight_scale,
        bias=linear.bias,
        out_dtype=linear.out_dtype,
        use_fast_accum=linear.fast_accum,
    )


def _fused_block_forward(block, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    batch, rows, cols = hidden_states.shape
    if batch != 1:
        raise ValueError(f"Fused DiT block expects batch=1, got shape {tuple(hidden_states.shape)}")
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(temb)
    h = hidden_states.reshape(rows, cols)
    idx, stride_tab, device = adaln_indices, shift_msa.stride(0), h.device
    attn = block.attn
    heads, dim = attn.heads, attn.head_dim
    q8 = torch.empty((rows, cols), device=device, dtype=FP8)
    s = torch.empty((rows, 1), device=device, dtype=torch.float32)

    _row_launch(
        _rms_mod_fp8_kernel,
        rows,
        cols,
        h,
        block.norm1.weight,
        scale_msa,
        shift_msa,
        idx,
        q8,
        s,
        cols,
        h.stride(0),
        stride_tab,
        block.norm1.eps,
    )
    qkv = _fp8_gemm(q8, s, block._fused_qkv)
    q = torch.empty((rows, heads, dim), device=device, dtype=h.dtype)
    k = torch.empty_like(q)
    cos, sin = rotary_emb
    _qk_norm_rope_kernel[(rows, 2 * heads)](
        qkv,
        attn.norm_q.weight,
        attn.norm_k.weight,
        cos,
        sin,
        q,
        k,
        qkv.stride(0),
        cos.stride(0),
        heads,
        attn.norm_q.eps,
        D=dim,
        ROT=cos.shape[-1],
        num_warps=2,
    )
    v = qkv[:, 2 * heads * dim :].view(rows, heads, dim)
    o = dispatch_attention_fn(
        q[None],
        k[None],
        v[None],
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        backend=attn.processor._attention_backend,
        parallel_config=attn.processor._parallel_config,
    )
    attn_out = attn.to_out[0](o.reshape(rows, heads * dim))
    h2 = torch.empty_like(h)
    _row_launch(
        _res_gate_rms_mod_fp8_kernel,
        rows,
        cols,
        h,
        attn_out,
        gate_msa,
        block.norm2.weight,
        scale_mlp,
        shift_mlp,
        idx,
        h2,
        q8,
        s,
        cols,
        h.stride(0),
        attn_out.stride(0),
        stride_tab,
        block.norm2.eps,
    )
    proj = _fp8_gemm(q8, s, block.ff.net[0].proj)
    inner = proj.shape[1] // 2
    f8 = torch.empty((rows, inner), device=device, dtype=FP8)
    _row_launch(_swiglu_fp8_kernel, rows, inner, proj, f8, s, inner, proj.stride(0))
    ff_out = _fp8_gemm(f8, s, block.ff.net[2])
    out = torch.empty_like(h)
    _row_launch(
        _res_gate_kernel,
        rows,
        cols,
        h2,
        ff_out,
        gate_mlp,
        idx,
        out,
        cols,
        h2.stride(0),
        ff_out.stride(0),
        stride_tab,
    )
    return out.view(batch, rows, cols)


@torch.no_grad()
def fuse_dit_blocks(transformer: nn.Module) -> None:
    """Fuse Q/K/V projections and replace each block's forward with `_fused_block_forward`."""
    for block in transformer.transformer_blocks:
        attn = block.attn
        parts = [attn.to_q, attn.to_k, attn.to_v]
        fused = FP8Linear.__new__(FP8Linear)
        nn.Module.__init__(fused)
        fused.register_buffer("weight", torch.cat([p.weight for p in parts]))
        fused.register_buffer("weight_scale", torch.cat([p.weight_scale for p in parts], dim=1))
        fused.bias = None
        fused.out_dtype = parts[0].out_dtype
        fused.fast_accum = parts[0].fast_accum
        fused.in_features = parts[0].in_features
        fused.out_features = sum(p.out_features for p in parts)
        block._fused_qkv = fused
        del attn.to_q, attn.to_k, attn.to_v
        block.forward = types.MethodType(_fused_block_forward, block)
    _free_cache()


# Batched VAE tile decoding --------------------------------------------------------------------------------------------


def batch_vae_tiles(vae, max_tiles: int = 48) -> None:
    """Batch identically shaped spatial tiles inside `AutoencoderKLMiniMaxH3Image._decode`."""

    def decode_tiled(z: torch.Tensor) -> torch.Tensor:
        if not vae.use_tiling:
            return vae.decoder(vae.post_quant_conv(z))
        ratio = vae.spatial_compression_ratio
        height = z.shape[-2] * ratio
        width = z.shape[-1] * ratio
        y_indices, y_lengths, y_overlaps = vae._split_tiles(
            height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = vae._split_tiles(
            width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width
        )
        tiles: dict[tuple[int, int], torch.Tensor] = {}
        for i, (i_pos, i_len) in enumerate(zip(y_indices, y_lengths)):
            for j, (j_pos, j_len) in enumerate(zip(x_indices, x_lengths)):
                tiles[(i, j)] = z[
                    ...,
                    i_pos // ratio : i_pos // ratio + i_len // ratio,
                    j_pos // ratio : j_pos // ratio + j_len // ratio,
                ]
        decoded: dict[tuple[int, int], torch.Tensor] = {}
        by_shape: dict[tuple[int, ...], list[tuple[int, int]]] = {}
        for key, tile in tiles.items():
            by_shape.setdefault(tuple(tile.shape), []).append(key)
        for keys in by_shape.values():
            for start in range(0, len(keys), max_tiles):
                chunk = keys[start : start + max_tiles]
                out = vae.decoder(vae.post_quant_conv(torch.cat([tiles[k] for k in chunk], dim=0)))
                for idx, key in enumerate(chunk):
                    decoded[key] = out[idx : idx + 1]
        rows = [[decoded[(i, j)] for j in range(len(x_indices))] for i in range(len(y_indices))]
        return vae._stitch_tiles(rows, y_overlaps, x_overlaps)

    vae._decode = decode_tiled


# Staged pipeline loader -----------------------------------------------------------------------------------------------


def load_fast_pipeline(
    repo_id: str,
    lora_repo: str | None = None,
    lora_filename: str | None = None,
    lora_revision: str = "main",
    lora_scale: float = 0.6,
    lora_attn_scale: float = 0.5,
    lora_mlp_scale: float = 0.9,
    lora_early_blocks: int = 10,
    lora_early_scale: float = 0.4,
    use_fast_kernels: bool = True,
    device: str = "cuda",
):
    """Load MiniMax-H3-Image without `text_encoder`, merge LoRA, and optionally install FP8 + fused DiT kernels."""
    from diffusers import ModularPipeline

    t0 = time.time()
    blocks = ModularPipeline.from_pretrained(repo_id).blocks
    blocks.sub_blocks.pop("text_encoder")
    pipe = blocks.init_pipeline(repo_id)

    # Load transformer alone first so AdaLN pruning + FP8 shrink RAM/VRAM before loading the VAE.
    pipe.load_components(names=["transformer"], dtype=torch.bfloat16)

    lora_entries = []
    if lora_repo and lora_filename:
        from huggingface_hub import hf_hub_download

        lora_path = hf_hub_download(repo_id=lora_repo, filename=lora_filename, revision=lora_revision)
        lora_entries = load_lora_entries(pipe.transformer, lora_path)
        merged = apply_lora_multipliers(
            lora_entries,
            scale=lora_scale,
            attn_scale=lora_attn_scale,
            mlp_scale=lora_mlp_scale,
            early_blocks=lora_early_blocks,
            early_scale=lora_early_scale,
        )
        print(
            f"[{time.time() - t0:5.1f}s] merged LoRA {lora_repo}/{lora_filename} "
            f"({len(lora_entries)} entries, {merged} active, scale={lora_scale:g}, "
            f"attn={lora_attn_scale:g}, mlp={lora_mlp_scale:g}, early[:{lora_early_blocks}]={lora_early_scale:g})",
            flush=True,
        )

    if use_fast_kernels:
        lora_entries.clear()
        pipe.load_components(names=["scheduler"])
        freed = install_adaln_tables(pipe.transformer, build_adaln_calls(pipe))
        q_count = quantize_fp8(pipe.transformer.transformer_blocks)
        fuse_dit_blocks(pipe.transformer)
        gc.collect()
        print(
            f"[{time.time() - t0:5.1f}s] fast kernels installed "
            f"(AdaLN freed {freed / 1e9:.2f}B params, {q_count} FP8 linears)",
            flush=True,
        )

    pipe.transformer.to(device)
    gc.collect()

    pipe.load_components(dtype=torch.bfloat16)
    batch_vae_tiles(pipe.vae)
    pipe.to(device)
    gc.collect()
    print(f"[{time.time() - t0:5.1f}s] pipeline ready on {device} (fast_kernels={use_fast_kernels})", flush=True)
    return pipe, lora_entries
