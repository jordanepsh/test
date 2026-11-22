---
title: MiniMax-H3-Image MysticXXX-V5 Fast
emoji: ⚡
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
python_version: "3.12"
short_description: Fast FP8+Triton MiniMax-H3-Image with MysticXXX-V5 LoRA
startup_duration_timeout: 1h
---

MiniMax-H3-Image (`MiniMaxAI/MiniMax-H3-Image`) through the diffusers modular pipeline,
preloaded with the converted Diffusers LoRA (`baka999/H3/MysticXXX_MMH3-V5_Image_diffusers_bf16.safetensors`,
Combo 2 defaults: scale `0.6`, attn `0.5`, mlp `0.9`, early blocks `0..9` scale `0.4`).

Fast quota-saving edition & public dependency adaptations:
- Fast inference (`H3_IMAGE_FAST_KERNELS=1` default): precomputed per-step AdaLN tables, rowwise FP8 W8A8 linears, fused QKV + Triton kernels, batched VAE tile decoding, and startup `_warmup` (`H3_IMAGE_WARMUP=1`) running on ZeroGPU `size="large"` (`48 GB`, `1×` quota, `~24 GB` peak VRAM). Set `H3_IMAGE_FAST_KERNELS=0` in Space variables for live BF16 LoRA sliders on `size="xlarge"`.
- ZeroGPU pool override: `H3_IMAGE_GPU_SIZE` (defaults to `large` when fast kernels are enabled, else `xlarge`)
- Output format: fast JPEG (`quality=95`) by default, with optional lossless PNG in **Advanced**
- Model repo: `MiniMaxAI/MiniMax-H3-Image` (override with `H3_IMAGE_REPO`)
- Qwen3-VL conditioner & Prompt Enhancement: `multimodalart/qwen3vl-conditioner` (override with `H3_IMAGE_ENCODER`; checking `Enhance prompt` uses `rewrite_prompt=True` when `H3_IMAGE_PE` is unset)
- Optional external prompt enhancer: `H3_IMAGE_PE` (takes precedence for text-to-image if configured)
- LoRA settings: `H3_IMAGE_LORA_REPO`, `H3_IMAGE_LORA_FILENAME`, `H3_IMAGE_LORA_SCALE`, `H3_IMAGE_LORA_ATTN_SCALE`, `H3_IMAGE_LORA_MLP_SCALE`, `H3_IMAGE_LORA_EARLY_BLOCKS`, `H3_IMAGE_LORA_EARLY_SCALE`
