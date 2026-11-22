import glob
import importlib.metadata
import logging
import os
import subprocess
import sys

# diffusers from the MiniMax-H3-Image PR branch (private repo, so the wheel ships with the Space).
try:
    _has_mm_diffusers = importlib.metadata.version("diffusers") == "0.42.0.dev0"
except importlib.metadata.PackageNotFoundError:
    _has_mm_diffusers = False

if not _has_mm_diffusers:
    for _wheel in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "diffusers-*.whl")):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", "--no-cache-dir", _wheel])

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import spaces  # before torch

import json
import math
import random
import tempfile
import time
from functools import cache, lru_cache

import gradio as gr
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision.io import encode_jpeg

from fast_mm_h3 import load_fast_pipeline


class _IgnoreRMSNormKernelWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "No kernel mapping found for layer" not in record.getMessage()


for _logger_name in ("kernels", "kernels.layer.layer"):
    logging.getLogger(_logger_name).addFilter(_IgnoreRMSNormKernelWarning())


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "off")


REPO = os.environ.get("H3_IMAGE_REPO", "MiniMaxAI/MiniMax-H3-Image").strip()
ENCODER_SPACE = os.environ.get("H3_IMAGE_ENCODER", "multimodalart/qwen3vl-conditioner").strip()
PE_SPACE = os.environ.get("H3_IMAGE_PE", "").strip()
FAST_KERNELS = _env_flag("H3_IMAGE_FAST_KERNELS", "0")
GPU_SIZE = os.environ.get("H3_IMAGE_GPU_SIZE", "").strip() or ("large" if FAST_KERNELS else "xlarge")
WARMUP = _env_flag("H3_IMAGE_WARMUP", "1")
DEFAULT_OUTPUT_FORMAT = "JPEG (fast)" if FAST_KERNELS else "PNG (lossless)"
# qwen3vl-conditioner's rewrite_prompt emits multi-second video/audio H3-Context-IR scripts (`[Shot 1]`,
# `overall_soundscape`, `non_diegetic_music`), which mismatch still-image MiniMax-H3-Image and burn ~120s quota.
CONDITIONER_REWRITE_ENABLED = "qwen3vl-conditioner" in ENCODER_SPACE and _env_flag(
    "H3_IMAGE_CONDITIONER_REWRITE", "0"
)
PE_ENABLED = bool(PE_SPACE or CONDITIONER_REWRITE_ENABLED)

DEFAULT_CONDITIONER_CANVAS = "960x544 · 16:9 fast"
DEFAULT_CONDITIONER_FRAMES = 56  # >= MIN_DURATION (2s) on multimodalart/qwen3vl-conditioner
MAX_IMAGES = 8
MAX_SEED = 2**31 - 1
MULTIPLE = 32

RATIOS = ["1:1", "5:4", "4:3", "3:2", "16:9", "21:9", "4:5", "3:4", "2:3", "9:16"]  # MiniMax-H3-Image-PE's set
MEGAPIXELS = {"4 MP": 4, "8 MP": 8, "16 MP": 16}
AUTO = "auto"  # PE's recommendation for text-to-image, Picture 1's ratio for edits, else 1:1

LORA_REPO = os.environ.get("H3_IMAGE_LORA_REPO", "baka999/H3").strip()
LORA_FILENAME = os.environ.get("H3_IMAGE_LORA_FILENAME", "MysticXXX_MMH3-V5_Image_diffusers_bf16.safetensors").strip()
LORA_REVISION = os.environ.get("H3_IMAGE_LORA_REVISION", "main").strip() or "main"
DEFAULT_LORA_SCALE = float(os.environ.get("H3_IMAGE_LORA_SCALE", "0.6"))
DEFAULT_LORA_ATTN_SCALE = float(os.environ.get("H3_IMAGE_LORA_ATTN_SCALE", "0.5"))
DEFAULT_LORA_MLP_SCALE = float(os.environ.get("H3_IMAGE_LORA_MLP_SCALE", "0.9"))
LORA_EARLY_BLOCKS = int(os.environ.get("H3_IMAGE_LORA_EARLY_BLOCKS", "10"))
DEFAULT_LORA_EARLY_SCALE = float(os.environ.get("H3_IMAGE_LORA_EARLY_SCALE", "0.4"))

# Everything but the Qwen3-VL text encoder, which runs on ENCODER_SPACE: `prompt_embeds` + `text_token_tags` become
# inputs of the pipeline.
pipe, LORA_ENTRIES = load_fast_pipeline(
    REPO,
    lora_repo=LORA_REPO or None,
    lora_filename=LORA_FILENAME or None,
    lora_revision=LORA_REVISION,
    lora_scale=DEFAULT_LORA_SCALE,
    lora_attn_scale=DEFAULT_LORA_ATTN_SCALE,
    lora_mlp_scale=DEFAULT_LORA_MLP_SCALE,
    lora_early_blocks=LORA_EARLY_BLOCKS,
    lora_early_scale=DEFAULT_LORA_EARLY_SCALE,
    use_fast_kernels=FAST_KERNELS,
    device="cuda",
)


EXTRA_LORA_REPO = os.environ.get("H3_EXTRA_LORA_REPO", "ghjjhv/kira-h3-loras-split").strip()
EXTRA_LORA_NAMES = [x for x in os.environ.get("H3_EXTRA_LORA_NAMES", "nb,slider,fn,large,skinny,bal").split(",") if x.strip()]
from lora_merge import apply_lora_multipliers
EXTRA_SETS = {}
if EXTRA_LORA_REPO and not FAST_KERNELS:
    from huggingface_hub import hf_hub_download as _hf_dl
    from lora_merge import load_lora_entries as _lle
    for _nm in EXTRA_LORA_NAMES:
        try:
            _p = _hf_dl(repo_id=EXTRA_LORA_REPO, filename=f"d_{_nm}.safetensors")
            EXTRA_SETS[_nm] = _lle(pipe.transformer, _p)
            print(f"loaded extra LoRA {_nm} ({len(EXTRA_SETS[_nm])} entries)", flush=True)
        except Exception as _e:
            print(f"extra LoRA {_nm} skipped: {_e!r}", flush=True)


def _apply_extra(extra):
    want = {}
    for part in (extra or "").split(","):
        part = part.strip()
        if ":" in part:
            nm, w = part.split(":", 1)
            try:
                want[nm.strip()] = float(w)
            except ValueError:
                pass
    for nm, ent in EXTRA_SETS.items():
        apply_lora_multipliers(ent, scale=want.get(nm, 0.0), attn_scale=1.0, mlp_scale=1.0, early_blocks=0, early_scale=1.0)


def _unlink_quietly(paths) -> None:
    for path in paths or ():
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def _extract_ip_token(request: gr.Request | None) -> str | None:
    headers = getattr(request, "headers", None) if request is not None else None
    return headers.get("x-ip-token") if headers else None


@lru_cache(maxsize=32)
def encoder(ip_token: str | None = None):
    from gradio_client import Client

    kwargs = {"token": os.environ.get("HF_TOKEN"), "httpx_kwargs": {"timeout": 600}}
    if ip_token:
        kwargs["headers"] = {"x-ip-token": ip_token}
    return Client(ENCODER_SPACE, **kwargs)


def _encode_with_conditioner(
    client,
    prompt: str,
    paths: list[str],
    encoder_space: str,
    handle_file_fn,
    enhance: bool = False,
) -> tuple[str, str | None]:
    """Send text or image-edit prompts to the configured remote Qwen3-VL conditioner Space."""
    if "qwen3vl-conditioner" in encoder_space:
        if paths:
            res = client.predict(
                prompt=prompt,
                media=[handle_file_fn(p) for p in paths],
                kinds=",".join(["image"] * len(paths)),
                canvas=DEFAULT_CONDITIONER_CANVAS,
                num_frames=DEFAULT_CONDITIONER_FRAMES,
                rewrite_prompt=bool(enhance),
                api_name="/encode_ref2va",
            )
        else:
            res = client.predict(
                prompt=prompt,
                image_path=None,
                last_image_path=None,
                canvas=DEFAULT_CONDITIONER_CANVAS,
                num_frames=DEFAULT_CONDITIONER_FRAMES,
                rewrite_prompt=bool(enhance),
                api_name="/encode",
            )
    else:
        res = client.predict(prompt, [{"image": handle_file_fn(p), "caption": None} for p in paths], api_name="/encode")
    wire_path = res[0] if isinstance(res, (list, tuple)) else res
    refined = None
    if enhance and isinstance(res, (list, tuple)) and len(res) > 1 and isinstance(res[1], dict):
        refined = (res[1].get("refined_prompt") or "").strip() or None
    return wire_path, refined


def image_paths(images):
    return [item[0] if isinstance(item, (list, tuple)) else item for item in images or []]


def size_at_area(width, height, pixels):
    scale = math.sqrt(pixels / (width * height))
    return (
        max(MULTIPLE, round(width * scale / MULTIPLE) * MULTIPLE),
        max(MULTIPLE, round(height * scale / MULTIPLE) * MULTIPLE),
    )


def resolve_size(paths, aspect, resolution):
    pixels = MEGAPIXELS.get(resolution, 4) * 1024 * 1024
    if aspect not in RATIOS:
        if paths:
            with Image.open(paths[0]) as im:
                width, height = im.size
            return size_at_area(width, height, pixels)
        aspect = "1:1"
    rw, rh = (int(v) for v in aspect.split(":"))
    return size_at_area(rw, rh, pixels)


@cache
def _cached_pe_client(space_id: str):
    from gradio_client import Client

    return Client(space_id, token=os.environ.get("HF_TOKEN"), httpx_kwargs={"timeout": 900})


def pe_client():
    if not PE_SPACE:
        raise gr.Error("Prompt enhancer Space (H3_IMAGE_PE) is not configured.")
    return _cached_pe_client(PE_SPACE)


def enhance_stream(prompt, effort, seed):
    """Streams MiniMax-H3-Image-PE: yields (thinking, None, None) while it reasons, then (thinking, caption, ratio)."""
    job = pe_client().submit(prompt, effort, int(seed) % (2**31), api_name="/expand")
    shown = 0
    for update in job:
        thinking = update[2] or ""
        if len(thinking) - shown > 200:
            shown = len(thinking)
            yield thinking, None, None
    caption, ratio, thinking, _ = job.result()
    yield thinking, caption, ratio


def prepare(
    prompt,
    images,
    aspect,
    resolution,
    seed,
    randomize,
    enhance: bool = False,
    ip_token: str | None = None,
):
    """Validate the request, pick the size and seed, and encode (optionally rewriting) on the conditioner Space."""
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt.")
    paths = image_paths(images)
    if len(paths) > MAX_IMAGES:
        raise gr.Error(f"At most {MAX_IMAGES} images.")
    for path in paths:
        with Image.open(path) as im:
            w, h = im.size
        if min(w, h) < MULTIPLE:
            raise gr.Error(f"Each reference image needs at least {MULTIPLE} pixels per side, got {w}x{h}.")
    width, height = resolve_size(paths, aspect, resolution)
    if randomize:
        seed = random.randint(0, MAX_SEED)
    from gradio_client import handle_file

    embeds_path, refined = _encode_with_conditioner(
        encoder(ip_token=ip_token),
        prompt.strip(),
        paths,
        ENCODER_SPACE,
        handle_file,
        enhance=enhance,
    )
    return embeds_path, int(seed), int(width), int(height), refined


def _duration(embeds_path, images, width, height, seed, *_, **__):
    # The canvas is at least 2048x2048; references add their own latent rows.
    megapixels = max(int(width) * int(height), 2048 * 2048) / 1e6
    for path in image_paths(images):
        with Image.open(path) as im:
            w, h = im.size
        megapixels += min(max(w * h / 1e6, 1.0), 4.2)
    if FAST_KERNELS:
        return min(300, int(20 + 2.5 * megapixels + 0.06 * megapixels**2))
    return min(300, int(30 + 4 * megapixels + 0.15 * megapixels**2))


def _save_output(images_pt: torch.Tensor, output_format: str = DEFAULT_OUTPUT_FORMAT) -> str:
    """Save `(1, 3, H, W)` or `(3, H, W)` float `[0, 1]` tensor as fast JPEG (quality 95) or lossless PNG."""
    img = images_pt[0] if images_pt.ndim == 4 else images_pt
    pixels = (img.clamp(0, 1) * 255).round().to(torch.uint8).contiguous()
    if "png" in str(output_format).lower():
        arr = pixels.permute(1, 2, 0).cpu().numpy()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            path = handle.name
        Image.fromarray(arr).save(path, format="PNG")
        return path
    try:
        data = encode_jpeg(pixels, quality=95)
    except Exception:  # noqa: BLE001 - CPU fallback when nvjpeg is unavailable
        data = encode_jpeg(pixels.cpu(), quality=95)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
        handle.write(data.cpu().numpy().tobytes())
        return handle.name


def _run(
    embeds_path,
    images,
    width,
    height,
    seed,
    output_format=DEFAULT_OUTPUT_FORMAT,
    lora_scale=DEFAULT_LORA_SCALE,
    lora_attn_scale=DEFAULT_LORA_ATTN_SCALE,
    lora_mlp_scale=DEFAULT_LORA_MLP_SCALE,
    lora_early_scale=DEFAULT_LORA_EARLY_SCALE,
    extra="",
):
    if LORA_ENTRIES:
        apply_lora_multipliers(
            LORA_ENTRIES,
            scale=float(lora_scale),
            attn_scale=float(lora_attn_scale),
            mlp_scale=float(lora_mlp_scale),
            early_blocks=LORA_EARLY_BLOCKS,
            early_scale=float(lora_early_scale),
        )
    _apply_extra(extra)
    embeds = load_file(embeds_path)
    kwargs = dict(
        prompt_embeds=embeds["prompt_embeds"].to("cuda", torch.bfloat16),
        text_token_tags=embeds["text_token_tags"],
        width=int(width),
        height=int(height),
        generator=torch.Generator("cpu").manual_seed(int(seed)),
        output_type="pt",
        output="images",
    )
    paths = image_paths(images)
    if paths:
        loaded = []
        for p in paths:
            with Image.open(p) as im:
                loaded.append(im.convert("RGB"))
        kwargs["image"] = loaded
    images_pt = pipe(**kwargs)
    return _save_output(images_pt, output_format)


@spaces.GPU(duration=_duration, size=GPU_SIZE)
def _generate(
    embeds_path,
    images,
    width,
    height,
    seed,
    output_format=DEFAULT_OUTPUT_FORMAT,
    lora_scale=DEFAULT_LORA_SCALE,
    lora_attn_scale=DEFAULT_LORA_ATTN_SCALE,
    lora_mlp_scale=DEFAULT_LORA_MLP_SCALE,
    lora_early_scale=DEFAULT_LORA_EARLY_SCALE,
    extra="",
):
    return _run(
        embeds_path,
        images,
        width,
        height,
        seed,
        output_format=output_format,
        lora_scale=lora_scale,
        lora_attn_scale=lora_attn_scale,
        lora_mlp_scale=lora_mlp_scale,
        lora_early_scale=lora_early_scale,
        extra=extra,
    )


@spaces.GPU(duration=120, size=GPU_SIZE)
def _warmup():
    """Compile and autotune Triton + CUDA kernels once at Space startup."""
    text_dim = int(getattr(pipe.transformer.config, "text_dim", 5120))
    prompt_embeds = torch.zeros((1, 16, text_dim), device="cuda", dtype=torch.bfloat16)
    text_token_tags = torch.ones((16,), dtype=torch.long)
    dummy = Image.new("RGB", (1024, 1024), (120, 140, 160))
    common = dict(
        prompt_embeds=prompt_embeds,
        text_token_tags=text_token_tags,
        width=1024,
        height=1024,
        min_generation_area=1024,
        generator=torch.Generator("cpu").manual_seed(0),
        output_type="pt",
        output="images",
    )
    t0 = time.time()
    pipe(**common)
    pipe(**common, image=[dummy])
    print(f"Warmup complete on size={GPU_SIZE} in {time.time() - t0:.1f}s", flush=True)


if WARMUP:
    try:
        _warmup()
    except Exception as error:  # noqa: BLE001 - Space still starts cleanly if warmup hits quota
        print(f"Warmup skipped: {error!r}", flush=True)


def generate(
    prompt,
    images=None,
    aspect=AUTO,
    resolution="4 MP",
    seed=0,
    randomize=True,
    enhance=False,
    reasoning_effort="low",
    lora_scale=DEFAULT_LORA_SCALE,
    lora_attn_scale=DEFAULT_LORA_ATTN_SCALE,
    lora_mlp_scale=DEFAULT_LORA_MLP_SCALE,
    lora_early_scale=DEFAULT_LORA_EARLY_SCALE,
    output_format=DEFAULT_OUTPUT_FORMAT,
    extra="",
    progress=gr.Progress(track_tqdm=True),
    request: gr.Request = None,
):
    """Text-to-image, or an edit of up to 8 images referred to as `<Picture 1>`, `<Picture 2>`, ... in the prompt.

    `aspect`: one of 1:1, 5:4, 4:3, 3:2, 16:9, 21:9, 4:5, 3:4, 2:3, 9:16, or "auto". `enhance`: expand the prompt via
    MiniMax-H3-Image-PE (when `H3_IMAGE_PE` is set) or Qwen3-VL conditioner (`rewrite_prompt=True`). Returns
    (image, seed, prompt used).
    """
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt.")
    final = prompt.strip()
    paths = image_paths(images)
    encode_enhance = False
    if enhance:
        if PE_SPACE and not paths:
            progress(0, desc="Enhancing the prompt ...")
            *_, (thinking, caption, ratio) = enhance_stream(final, reasoning_effort, seed)
            final = caption or final
            if aspect == AUTO and ratio:
                aspect = ratio
        elif CONDITIONER_REWRITE_ENABLED:
            encode_enhance = True
    progress(0.05, desc="Enhancing & encoding prompt ..." if encode_enhance else "Encoding the prompt ...")
    embeds_path, seed, width, height, refined = prepare(
        final,
        images,
        aspect,
        resolution,
        seed,
        randomize,
        enhance=encode_enhance,
        ip_token=_extract_ip_token(request),
    )
    if refined:
        final = refined
    try:
        progress(0.1, desc=f"Generating {width}x{height} ...")
        out_path = _generate(
            embeds_path,
            images,
            width,
            height,
            seed,
            output_format,
            float(lora_scale),
            float(lora_attn_scale),
            float(lora_mlp_scale),
            float(lora_early_scale),
            extra,
        )
    finally:
        _unlink_quietly([embeds_path])
    return out_path, seed, final


def studio(
    state,
    seed,
    randomize,
    effort,
    lora_scale,
    lora_attn_scale,
    lora_mlp_scale,
    lora_early_scale,
    output_format,
    request: gr.Request = None,
):
    """The composer's Generate, part 1: optional prompt enhancement and encoding. Returns the render plan."""
    state = state or {}
    prompt = (state.get("prompt") or "").strip()
    aspect = state.get("aspect") or AUTO
    resolution = state.get("resolution") or "4 MP"
    if not prompt:
        raise gr.Error("Describe the image first.")
    paths = decode_pictures(state.get("images"))
    if randomize:
        seed = random.randint(0, MAX_SEED)
    final, thinking, encode_enhance = prompt, "", False
    try:
        if state.get("enhance"):
            if PE_SPACE and not paths:
                yield "✨ Enhancing the prompt with MiniMax-H3-Image-PE ...", gr.update(), "", seed, None
                t0 = time.time()
                caption = ratio = None
                for thinking, caption, ratio in enhance_stream(prompt, effort, seed):
                    if caption is None:
                        yield (
                            f"✨ Enhancing the prompt · thinking ({len(thinking.split())} words) ...",
                            gr.update(),
                            thinking,
                            seed,
                            None,
                        )
                if caption:
                    final = caption
                    if aspect == AUTO and ratio:
                        aspect = ratio
                yield f"✨ Prompt enhanced in {time.time() - t0:.0f}s · encoding ...", final, thinking, seed, None
            elif CONDITIONER_REWRITE_ENABLED:
                encode_enhance = True
            elif PE_SPACE and paths:
                gr.Info("Prompt enhancement on H3_IMAGE_PE is text-to-image only; edits use your prompt as written.")
            else:
                gr.Info(
                    "Prompt enhancement is disabled (set H3_IMAGE_PE for still-image PE; "
                    "qwen3vl-conditioner video/audio rewrite is off by default)."
                )
        width, height = resolve_size(paths, aspect, resolution)
        status = (
            f"✨ Enhancing & encoding prompt with Qwen3-VL conditioner · {width}×{height} ..."
            if encode_enhance
            else f"Encoding the prompt · {width}×{height} ..."
        )
        yield status, final, thinking, seed, None
        embeds_path, seed, width, height, refined = prepare(
            final,
            paths,
            aspect,
            resolution,
            seed,
            False,
            enhance=encode_enhance,
            ip_token=_extract_ip_token(request),
        )
        if refined:
            final = refined
    except Exception:
        _unlink_quietly(paths)
        raise
    plan = {
        "embeds": embeds_path,
        "images": paths,
        "tmp_paths": list(paths),
        "width": width,
        "height": height,
        "seed": seed,
        "output_format": output_format or DEFAULT_OUTPUT_FORMAT,
        "lora": {
            "lora_scale": float(lora_scale),
            "lora_attn_scale": float(lora_attn_scale),
            "lora_mlp_scale": float(lora_mlp_scale),
            "lora_early_scale": float(lora_early_scale),
        },
    }
    yield f"Generating {width}×{height} ...", final, thinking, seed, plan


def render(plan, progress=gr.Progress(track_tqdm=True)):
    """Part 2: generate the image on ZeroGPU, cleaning temporary encode/upload files afterwards."""
    if not plan:
        raise gr.Error("Missing render plan — prompt preparation did not complete.")
    t0 = time.time()
    lora_cfg = plan.get("lora") or {}
    output_format = plan.get("output_format") or DEFAULT_OUTPUT_FORMAT
    try:
        path = _generate(
            plan["embeds"],
            plan["images"],
            plan["width"],
            plan["height"],
            plan["seed"],
            output_format,
            **lora_cfg,
        )
    finally:
        _unlink_quietly([plan.get("embeds"), *(plan.get("tmp_paths") or [])])
    mode_tag = f"FP8+Triton {GPU_SIZE}" if FAST_KERNELS else f"BF16 {GPU_SIZE}"
    stats = f"{plan['width']}×{plan['height']} · seed {plan['seed']} · {time.time() - t0:.1f}s ({mode_tag})"
    if LORA_ENTRIES and lora_cfg:
        stats += (
            f" · LoRA {lora_cfg.get('lora_scale', DEFAULT_LORA_SCALE):g} "
            f"(attn {lora_cfg.get('lora_attn_scale', DEFAULT_LORA_ATTN_SCALE):g}, "
            f"mlp {lora_cfg.get('lora_mlp_scale', DEFAULT_LORA_MLP_SCALE):g}, "
            f"early {lora_cfg.get('lora_early_scale', DEFAULT_LORA_EARLY_SCALE):g})"
        )
    return path, stats

IDEAS = [
    "a two-color risograph print of a fox reading a newspaper on a park bench, headline \"FOX NEWS\"",
    "a 1970s film frame of a woman waiting at a rainy Tokyo bus stop, lit by a vending machine",
    "isometric cutaway of a tiny underground bakery",
    "a poster for a jazz festival in Lisbon",
]

COMPOSER_DEFAULTS = {"prompt": "", "aspect": AUTO, "resolution": "4 MP", "enhance": False, "images": []}


def decode_pictures(urls):
    """The composer's pictures (data URLs, in <Picture i> order) as temp files."""
    import base64

    paths = []
    for url in (urls or [])[:MAX_IMAGES]:
        header, _, data = str(url).partition(",")
        ext = {"image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}.get(header[5:].split(";")[0], ".jpg")
        path = tempfile.NamedTemporaryFile(suffix=ext, delete=False).name
        with open(path, "wb") as f:
            f.write(base64.b64decode(data))
        paths.append(path)
    return paths

COMPOSER_HTML = """
<div class="hx-card">
  <div class="hx-head">
    <div class="hx-seg" data-role="resolution" role="tablist">
      <button type="button" class="hx-seg-btn" data-value="4 MP">4 MP</button>
      <button type="button" class="hx-seg-btn" data-value="8 MP">8 MP</button>
      <button type="button" class="hx-seg-btn" data-value="16 MP">16 MP</button>
    </div>
    <span class="hx-headhint">text-to-image &middot; or edit up to 8 pictures</span>
  </div>
  <div class="hx-pics" data-role="pics">
    <div class="hx-thumbs" data-role="thumbs"></div>
    <label class="hx-add" title="Optional: up to 8 pictures to edit; drop files here too">
      <input type="file" accept="image/*" multiple data-role="files" hidden>
      <span class="hx-tile" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
          <rect x="3" y="4" width="18" height="16" rx="2.5"/><circle cx="8.5" cy="9.5" r="1.7"/><path d="M3.5 17.5l5-5 4 4 2.5-2.5 5.5 5.5"/>
        </svg>
        <span class="hx-tile-plus">+</span>
      </span>
      <span class="hx-addtext"><span class="hx-addtitle">Pictures to edit</span><span class="hx-opt">optional &middot; refer to them as &lt;Picture 1&gt;, &lt;Picture 2&gt;&#8230;</span></span>
    </label>
  </div>
  <div class="hx-pictags" data-role="pictags" hidden><span class="hx-pictags-label">Insert</span></div>
  <textarea class="hx-desc" data-role="prompt" rows="4"
    placeholder="Describe your image&#8230;  e.g. a poster for a jazz festival in Lisbon, or an edit of the pictures above"></textarea>
  <div class="hx-chips" data-role="idea-chips"></div>
  <div class="hx-row">
    <span class="hx-label">Aspect</span>
    <div class="hx-ratios" data-role="aspect"></div>
  </div>
  <div class="hx-foot">
    <label class="hx-toggle" data-role="enhance-wrap" title="Expand the prompt into a detailed caption via MiniMax-H3-Image-PE (set H3_IMAGE_PE; qwen3vl-conditioner video/audio rewrite is off by default)">
      <input type="checkbox" data-role="enhance">&#10024; Enhance prompt
    </label>
    <span class="hx-spacer"></span>
    <button type="button" class="hx-primary" data-role="generate">Generate</button>
  </div>
</div>
"""

COMPOSER_CSS = """
.hx-card { background: var(--block-background-fill); border: var(--block-border-width, 1px) solid var(--block-border-color, var(--border-color-primary)); border-radius: var(--block-radius, 12px); box-shadow: var(--block-shadow, none); padding: 16px; display: flex; flex-direction: column; gap: 12px; }
.hx-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap; }
.hx-seg { display: inline-flex; background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary); border-radius: 999px; padding: 3px; gap: 2px; }
.hx-seg-btn { border: none; background: transparent; color: var(--body-text-color-subdued); padding: 5px 14px; border-radius: 999px; font-family: inherit; font-size: 13.5px; font-weight: 600; cursor: pointer; transition: background .15s, color .15s; }
.hx-seg-btn[aria-selected="true"] { background: var(--button-primary-background-fill); color: var(--button-primary-text-color); }
.hx-headhint { color: var(--body-text-color-subdued); font-size: 12.5px; }
.hx-card textarea { width: 100%; box-sizing: border-box; background: var(--input-background-fill); border: var(--input-border-width, 1px) solid var(--input-border-color, var(--border-color-primary)); border-radius: var(--input-radius, 8px); padding: 10px 12px; color: var(--body-text-color); font-family: inherit; font-size: 16px; line-height: 1.5; min-height: 118px; resize: vertical; transition: border-color .15s; }
.hx-card textarea::placeholder { color: var(--input-placeholder-color, var(--body-text-color-subdued)); }
.hx-card textarea:focus { outline: none; border-color: var(--input-border-color-focus, var(--color-accent)); }
.hx-pics { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; min-height: 72px; box-sizing: border-box; border: 1px dashed var(--border-color-primary); border-radius: var(--radius-lg, 8px); padding: 9px 10px; transition: border-color .15s, background .15s; }
.hx-pics.hx-drag { border-color: var(--color-accent); background: var(--background-fill-secondary); }
.hx-thumbs { display: flex; gap: 6px; flex-wrap: wrap; }
.hx-thumbs:empty { display: none; }
.hx-thumb { position: relative; width: 52px; height: 52px; border-radius: 6px; overflow: hidden; border: 1px solid var(--border-color-primary); }
.hx-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
.hx-thumb .hx-n { position: absolute; left: 2px; bottom: 2px; background: rgba(0,0,0,.65); color: #fff; font-size: 10px; line-height: 1; padding: 2px 4px; border-radius: 4px; }
.hx-thumb .hx-x { position: absolute; right: 1px; top: 1px; width: 16px; height: 16px; border: none; border-radius: 50%; background: rgba(0,0,0,.65); color: #fff; font-size: 11px; line-height: 16px; padding: 0; cursor: pointer; }
.hx-add { display: inline-flex; align-items: center; gap: 10px; color: var(--body-text-color-subdued); cursor: pointer; user-select: none; }
.hx-tile { position: relative; flex: none; width: 52px; height: 52px; box-sizing: border-box; display: flex; align-items: center; justify-content: center; border: 1px dashed var(--border-color-primary); border-radius: 8px; background: var(--background-fill-secondary); transition: color .15s, border-color .15s; }
.hx-tile svg { width: 28px; height: 28px; }
.hx-tile-plus { position: absolute; right: -6px; bottom: -6px; width: 18px; height: 18px; border-radius: 50%; background: var(--button-primary-background-fill); color: var(--button-primary-text-color); font-size: 14px; font-weight: 700; line-height: 18px; text-align: center; }
.hx-add:hover .hx-tile { color: var(--color-accent); border-color: var(--color-accent); }
.hx-addtext { display: flex; flex-direction: column; gap: 2px; }
.hx-addtitle { color: var(--body-text-color); font-size: 13.5px; font-weight: 600; }
.hx-add:hover .hx-addtitle { color: var(--color-accent); }
.hx-opt { font-size: 11.5px; opacity: .85; }
.hx-pictags { display: flex; align-items: center; gap: 5px; flex-wrap: wrap; margin-top: -4px; }
.hx-pictags[hidden] { display: none; }
.hx-pictags-label { color: var(--body-text-color-subdued); font-size: 12px; margin-right: 2px; }
.hx-pictag { background: transparent; color: var(--body-text-color-subdued); border: 1px dashed var(--border-color-primary); border-radius: 6px; padding: 2px 8px; font-family: var(--font-mono, ui-monospace, monospace); font-size: 11.5px; cursor: pointer; transition: color .15s, border-color .15s; }
.hx-pictag:hover { color: var(--color-accent); border-color: var(--color-accent); }
.hx-thumb img { cursor: pointer; }
.hx-chips { display: flex; flex-wrap: wrap; gap: 6px; }
.hx-chip { background: var(--button-secondary-background-fill); color: var(--button-secondary-text-color); border: 1px solid var(--button-secondary-border-color, var(--border-color-primary)); border-radius: 999px; padding: 4px 12px; font-family: inherit; font-size: 12.5px; cursor: pointer; transition: border-color .15s; text-align: left; }
.hx-chip:hover { border-color: var(--color-accent); }
.hx-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.hx-label { color: var(--block-title-text-color, var(--body-text-color)); font-size: 13px; font-weight: 600; }
.hx-ratios { display: flex; flex-wrap: wrap; gap: 4px; }
.hx-ratio { background: transparent; color: var(--body-text-color-subdued); border: 1px dashed var(--border-color-primary); border-radius: 6px; padding: 3px 9px; font-family: var(--font-mono, ui-monospace, monospace); font-size: 12px; cursor: pointer; transition: color .15s, border-color .15s, background .15s; }
.hx-ratio:hover { color: var(--color-accent); border-color: var(--color-accent); }
.hx-ratio[aria-selected="true"] { border-style: solid; border-color: var(--color-accent); color: var(--color-accent); background: var(--background-fill-secondary); }
.hx-foot { display: flex; align-items: center; gap: 10px; }
.hx-spacer { flex: 1; }
.hx-toggle { display: inline-flex; align-items: center; gap: 7px; color: var(--body-text-color); font-size: 14px; cursor: pointer; user-select: none; }
.hx-toggle input { width: 16px; height: 16px; accent-color: var(--color-accent); cursor: pointer; }
.hx-primary { background: var(--button-primary-background-fill); color: var(--button-primary-text-color); border: var(--button-border-width, 1px) solid var(--button-primary-border-color, transparent); border-radius: var(--button-large-radius, var(--radius-lg, 8px)); padding: 10px 28px; font-family: inherit; font-size: var(--button-large-text-size, 16px); font-weight: 600; cursor: pointer; box-shadow: var(--button-primary-shadow, none); transition: background .15s, transform .05s; }
.hx-primary:hover { background: var(--button-primary-background-fill-hover, var(--button-primary-background-fill)); }
.hx-primary:active { transform: translateY(1px); }
"""

COMPOSER_JS = """
const $ = function(s) { return element.querySelector(s); };
const $$ = function(s) { return Array.from(element.querySelectorAll(s)); };
const IDEAS = __IDEAS__;
const RATIOS = ['auto'].concat(__RATIOS__);
const PE_ENABLED = __PE_ENABLED__;
const prompt = $('[data-role="prompt"]');
const enhance = $('[data-role="enhance"]');
if (!PE_ENABLED) {
  enhance.checked = false;
  enhance.disabled = true;
  const wrap = $('[data-role="enhance-wrap"]');
  if (wrap) {
    wrap.title = 'Prompt enhancer disabled (set H3_IMAGE_PE for still-image PE; qwen3vl-conditioner video/audio rewrite is off by default)';
    wrap.style.opacity = '0.5';
  }
}
let aspect = 'auto', resolution = '4 MP', lastPushed = '', pics = [];
const MAX_PICS = 8;
function insertTag(tag) {
  const v = prompt.value;
  const at = prompt.selectionStart == null ? v.length : prompt.selectionStart;
  const end = prompt.selectionEnd == null ? at : prompt.selectionEnd;
  const before = v.slice(0, at), after = v.slice(end);
  const ins = (before && !/\\s$/.test(before) ? ' ' : '') + tag + (after && !/^\\s/.test(after) ? ' ' : (after ? '' : ' '));
  prompt.value = before + ins + after;
  const pos = (before + ins).length;
  prompt.focus();
  prompt.setSelectionRange(pos, pos);
}
function renderTags() {
  const row = $('[data-role="pictags"]');
  Array.from(row.querySelectorAll('.hx-pictag')).forEach(function(b) { b.remove(); });
  row.hidden = pics.length === 0;
  pics.forEach(function(_, i) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'hx-pictag'; b.textContent = '<Picture ' + (i + 1) + '>';
    b.title = 'Insert into the prompt';
    b.addEventListener('mousedown', function(e) { e.preventDefault(); });  // keep the prompt's cursor
    b.addEventListener('click', function() { insertTag('<Picture ' + (i + 1) + '>'); });
    row.appendChild(b);
  });
}
function renderPics() {
  renderTags();
  const box = $('[data-role="thumbs"]');
  box.innerHTML = '';
  pics.forEach(function(url, i) {
    const t = document.createElement('div'); t.className = 'hx-thumb';
    const img = document.createElement('img'); img.src = url; img.title = 'Insert <Picture ' + (i + 1) + '> into the prompt';
    img.addEventListener('click', function() { insertTag('<Picture ' + (i + 1) + '>'); });
    t.appendChild(img);
    const n = document.createElement('span'); n.className = 'hx-n'; n.textContent = i + 1; t.appendChild(n);
    const x = document.createElement('button'); x.type = 'button'; x.className = 'hx-x'; x.textContent = '\u00d7'; x.title = 'Remove';
    x.addEventListener('click', function() { pics.splice(i, 1); renderPics(); });
    t.appendChild(x);
    box.appendChild(t);
  });
}
function addFiles(files) {
  Array.from(files || []).filter(function(f) { return f.type.indexOf('image/') === 0; }).forEach(function(f) {
    if (pics.length >= MAX_PICS) return;
    const r = new FileReader();
    r.onload = function() { if (pics.length < MAX_PICS) { pics.push(r.result); renderPics(); } };
    r.readAsDataURL(f);
  });
}
const picsBox = $('[data-role="pics"]');
$('[data-role="files"]').addEventListener('change', function(e) { addFiles(e.target.files); e.target.value = ''; });
['dragenter', 'dragover'].forEach(function(ev) { picsBox.addEventListener(ev, function(e) { e.preventDefault(); picsBox.classList.add('hx-drag'); }); });
['dragleave', 'drop'].forEach(function(ev) { picsBox.addEventListener(ev, function(e) { e.preventDefault(); picsBox.classList.remove('hx-drag'); }); });
picsBox.addEventListener('drop', function(e) { addFiles(e.dataTransfer.files); });

RATIOS.forEach(function(r) {
  const b = document.createElement('button');
  b.type = 'button'; b.className = 'hx-ratio'; b.dataset.value = r;
  b.textContent = r === 'auto' ? 'Auto' : r;
  if (r === 'auto') b.title = 'Enhanced: the ratio PE recommends. Editing: Picture 1\\'s ratio. Otherwise 1:1.';
  b.addEventListener('click', function() { setAspect(r); });
  $('[data-role="aspect"]').appendChild(b);
});
IDEAS.forEach(function(t, i) {
  const b = document.createElement('button');
  b.type = 'button'; b.className = 'hx-chip'; b.textContent = t;
  b.addEventListener('click', function() { prompt.value = t; gather({example_key: 'idea_' + i}); trigger('apply'); });
  $('[data-role="idea-chips"]').appendChild(b);
});
function setAspect(r) { aspect = r; $$('.hx-ratio').forEach(function(b) { b.setAttribute('aria-selected', String(b.dataset.value === r)); }); }
function setResolution(r) { resolution = r; $$('.hx-seg-btn').forEach(function(b) { b.setAttribute('aria-selected', String(b.dataset.value === r)); }); }
$$('.hx-seg-btn').forEach(function(b) { b.addEventListener('click', function() { setResolution(b.dataset.value); }); });
function readState() { return { prompt: prompt.value, aspect: aspect, resolution: resolution, enhance: PE_ENABLED && enhance.checked, images: pics.slice() }; }
function gather(extra) { const s = Object.assign(readState(), extra || {}); lastPushed = JSON.stringify(s); props.value = s; }
function applyState(v) {
  if (!v) return;
  if (prompt.value !== (v.prompt || '')) prompt.value = v.prompt || '';
  enhance.checked = PE_ENABLED ? !!v.enhance : false;
  setAspect(v.aspect || 'auto');
  setResolution(v.resolution || '4 MP');
  if (Array.isArray(v.images)) { pics = v.images.slice(0, MAX_PICS); renderPics(); }
}
$('[data-role="generate"]').addEventListener('click', function() { gather(); trigger('submit'); });
prompt.addEventListener('keydown', function(e) { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); gather(); trigger('submit'); } });
watch('value', function() { if (JSON.stringify(props.value) !== lastPushed) applyState(props.value); });
applyState(props.value);
""".replace("__IDEAS__", json.dumps(IDEAS)).replace("__RATIOS__", json.dumps(RATIOS)).replace("__PE_ENABLED__", json.dumps(bool(PE_ENABLED)))

CSS = """
#col-container { max-width: 1300px; margin: 0 auto; }
.dark .gradio-container { color: var(--body-text-color); }
.html-container { padding: 0; }
#hx-composer, #hx-composer > div, #hx-composer .html-container { padding: 0 !important; }
.hx-brand { display: flex; align-items: center; justify-content: center; gap: 12px; margin: 12px auto 0; max-width: 320px; }
.hx-brand img { height: 40px; width: auto; max-width: 140px; object-fit: contain; display: block; }
.hx-logo-dark { display: none !important; }
.dark .hx-logo-light { display: none !important; }
.dark .hx-logo-dark { display: block !important; }
.hx-word { font-size: 30px; font-weight: 400; line-height: 1; letter-spacing: -0.01em; color: var(--body-text-color); white-space: nowrap; }
.hx-grad { background: linear-gradient(90deg, #e8197e, #ff6a3d); -webkit-background-clip: text; background-clip: text; color: transparent; font-weight: 500; }
#hx-links { text-align: center; }
"""

import base64

HERE = os.path.dirname(os.path.abspath(__file__))
LOGO_LIGHT = base64.b64encode(open(os.path.join(HERE, "minimax_logo_light.png"), "rb").read()).decode()
LOGO_DARK = base64.b64encode(open(os.path.join(HERE, "minimax_logo_dark.png"), "rb").read()).decode()
EXAMPLES = {}
if os.path.exists(os.path.join(HERE, "examples", "manifest.json")):
    EXAMPLES = json.load(open(os.path.join(HERE, "examples", "manifest.json"), encoding="utf-8"))


def load_example(state):
    """An idea chip: show its precomputed result (PE at low effort, then H3-Image at PE's ratio, 4 MP)."""
    ex = EXAMPLES.get((state or {}).get("example_key") or "")
    if not ex:
        return gr.update(), gr.update(), gr.update(), gr.update()
    stats = f"cached example · enhanced prompt · {ex['aspect']} · {ex['size']} · seed {ex['seed']}"
    return os.path.join(HERE, ex["image"]), stats, ex["prompt_used"], ex["thinking"]


with gr.Blocks(theme=gr.themes.Citrus(), css=CSS, title="MiniMax H3 Image · MysticXXX-V5 LoRA") as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(
            '<div class="hx-brand">'
            f'<img src="data:image/png;base64,{LOGO_LIGHT}" class="hx-logo-light" alt="MiniMax">'
            f'<img src="data:image/png;base64,{LOGO_DARK}" class="hx-logo-dark" alt="MiniMax">'
            '<span class="hx-word"><span class="hx-grad">H3</span> Image</span></div>',
            container=False, padding=False,
        )
        gr.Markdown(
            f"Text-to-image and editing of up to 8 pictures in 6 steps · "
            f"`{'FP8+Triton' if FAST_KERNELS else 'BF16'}` on `{GPU_SIZE}` · "
            f"LoRA `{LORA_FILENAME}` (Combo 2: scale `{DEFAULT_LORA_SCALE:g}`, "
            f"attn `{DEFAULT_LORA_ATTN_SCALE:g}`, mlp `{DEFAULT_LORA_MLP_SCALE:g}`, "
            f"early[:{LORA_EARLY_BLOCKS}] `{DEFAULT_LORA_EARLY_SCALE:g}`) · "
            f"Conditioner: `{ENCODER_SPACE}` "
            f"[[base model]](https://huggingface.co/{REPO}) | "
            f"[[LoRA]](https://huggingface.co/{LORA_REPO}) | "
            f"[[conditioner]](https://huggingface.co/spaces/{ENCODER_SPACE})",
            elem_id="hx-links",
        )
        with gr.Row():
            with gr.Column():
                composer = gr.HTML(
                    value=dict(COMPOSER_DEFAULTS),
                    elem_id="hx-composer",
                    html_template=COMPOSER_HTML,
                    css_template=COMPOSER_CSS,
                    js_on_load=COMPOSER_JS,
                    container=False,
                    padding=False,
                )
                images = gr.Gallery(type="filepath", visible=False)  # API input; the composer carries its own pictures
                with gr.Accordion("Advanced", open=False):
                    effort = gr.Radio(
                        ["low", "medium", "xhigh"],
                        value="low",
                        visible=bool(PE_SPACE),
                        label="Prompt enhancer reasoning effort",
                        info="xhigh is the released default; low thinks briefly and is several times faster",
                    )
                    output_format = gr.Radio(
                        ["JPEG (fast)", "PNG (lossless)"],
                        value=DEFAULT_OUTPUT_FORMAT,
                        label="Output format",
                        info="JPEG quality 95 encodes in ~20ms; PNG is lossless but slower on 8-16 MP",
                    )
                    if FAST_KERNELS:
                        gr.Markdown(
                            f"_Fast FP8+Triton mode active on ZeroGPU `{GPU_SIZE}` (`1×` quota): Combo 2 LoRA weights "
                            f"are baked before quantization. Set `H3_IMAGE_FAST_KERNELS=0` in Space variables "
                            f"for live BF16 sliders (`xlarge`)._"
                        )
                    with gr.Row():
                        lora_scale = gr.Slider(
                            label="LoRA scale",
                            minimum=0.0,
                            maximum=1.5,
                            step=0.05,
                            value=DEFAULT_LORA_SCALE,
                            interactive=bool(LORA_ENTRIES),
                        )
                        lora_attn_scale = gr.Slider(
                            label="Attention scale",
                            minimum=0.0,
                            maximum=1.5,
                            step=0.05,
                            value=DEFAULT_LORA_ATTN_SCALE,
                            interactive=bool(LORA_ENTRIES),
                        )
                    with gr.Row():
                        lora_mlp_scale = gr.Slider(
                            label="MLP (FFN) scale",
                            minimum=0.0,
                            maximum=1.5,
                            step=0.05,
                            value=DEFAULT_LORA_MLP_SCALE,
                            interactive=bool(LORA_ENTRIES),
                        )
                        lora_early_scale = gr.Slider(
                            label=f"Early blocks scale (0..{max(LORA_EARLY_BLOCKS - 1, 0)})",
                            minimum=0.0,
                            maximum=1.5,
                            step=0.05,
                            value=DEFAULT_LORA_EARLY_SCALE,
                            interactive=bool(LORA_ENTRIES),
                        )
                    with gr.Row():
                        seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                        randomize = gr.Checkbox(label="Randomize seed", value=True)
            with gr.Column():
                output = gr.Image(label="Result", type="filepath", show_label=False, height=620)
                stats = gr.Markdown()
                with gr.Accordion("Prompt used", open=False):
                    prompt_used = gr.Textbox(show_label=False, lines=8)
                with gr.Accordion("Enhancer thinking", open=False, visible=bool(PE_SPACE)):
                    thinking = gr.Textbox(show_label=False, lines=10)

    plan = gr.State()
    composer.submit(
        lambda: (None, None), None, [output, plan], show_progress="hidden", api_visibility="private",
    ).then(
        studio,
        [composer, seed, randomize, effort, lora_scale, lora_attn_scale, lora_mlp_scale, lora_early_scale, output_format],
        [stats, prompt_used, thinking, seed, plan],
        show_progress="minimal",
        api_visibility="private",
    ).success(
        render, [plan], [output, stats], show_progress="full", show_progress_on=[output], api_visibility="private",
    )

    composer.apply(
        load_example, [composer], [output, stats, prompt_used, thinking],
        show_progress="hidden", api_visibility="private",
    )

    # Plain API endpoint (also what the composer runs, minus the streaming UI updates).
    api_prompt = gr.Textbox(visible=False)
    api_aspect = gr.Textbox(visible=False, value=AUTO)
    api_resolution = gr.Textbox(visible=False, value="4 MP")
    api_enhance = gr.Checkbox(visible=False, value=False)
    api_effort = gr.Textbox(visible=False, value="low")
    api_extra = gr.Textbox(visible=False, value="")
    api_button = gr.Button(visible=False)
    api_button.click(
        generate,
        [
            api_prompt,
            images,
            api_aspect,
            api_resolution,
            seed,
            randomize,
            api_enhance,
            api_effort,
            lora_scale,
            lora_attn_scale,
            lora_mlp_scale,
            lora_early_scale,
            output_format,
            api_extra,
        ],
        [output, seed, prompt_used],
        api_name="generate",
    )

demo.launch()
