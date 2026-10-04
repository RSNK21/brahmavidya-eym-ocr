"""EyM Sanskrit Cloud HTR — Hugging Face Space (Gradio API backend).

Zero-install neural backend for the EyM Scholar's Workstation
(https://indragopa.github.io/eymocr/). The static site calls this
Space straight from the browser: no server on the scholar's computer.

Engines (same family as ocr_server.py):
  midf  → tadad/midf-sanskrit-ocr: Kraken BLLA line segmenter + PP-OCRv6
          recogniser, 17 MB total, CPU-only, no GPU and no quota. DEFAULT for
          line and word crops. It recognises a whole line at a time and emits
          no spaces, so word boxes are re-derived from the detected word gaps.
  qwen  → diabolic6045/Sanskrit-Qwen2.5-VL-7B-Instruct-OCR, 4-bit, needs GPU.
          On ZeroGPU hardware this runs inside @spaces.GPU. First-ever use
          downloads ~15GB, so Qwen loads LAZILY and MIDF covers for it. Best
          for whole pages and hard cases.
  trocr → paudelanil/trocr-devanagari-2 (+ optional DevGen LoRA), word-level,
          CPU-friendly. Loaded at startup; the reliable always-on fallback.
  auto  → MIDF (CPU, instant) for crops; Qwen only when explicitly asked.

API (Gradio /predict): inputs [image, engine_choice] → outputs [result_json],
where result_json = {"text":..., "lines":[{text,confidence,bbox,words?}...],
"engine":..., "medium":..., "note":...}. bboxes are page-normalized
{xmin,ymin,xmax,ymax}, matching the /ocr contract the frontend already parses.

Hardware: create the Space with **ZeroGPU**, though MIDF needs no GPU at all
(a plain CPU Space runs it in ~0.2 s per line). On CPU-only Spaces Qwen
auto-disables itself; MIDF and TrOCR still answer.
Env knobs: EYM_TROCR_ADAPTER_ID, EYM_QWEN_MODEL_ID, EYM_TROCR_MODEL_ID,
           EYM_SPACE_NO_QWEN=1 (skip the vision model, e.g. for local tests),
           EYM_MIDF_MODEL_ID, EYM_MIDF_MODE (line|page), EYM_MIDF=0 (disable),
           EYM_MIDF_TRIM_PAD, EYM_MIDF_MAX_H.
"""

import functools
import importlib.util
import io
import json
import os
import sys
import threading
import time
import unicodedata
import traceback

from PIL import Image as PILImage

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

import torch

# Heavy vision & TrOCR classes are imported lazily inside their respective
# loaders to keep startup memory under 200 MB.
AutoProcessor = None
VisionEncoderDecoderModel = None
TrOCRProcessor = None
Qwen2_5_VLForConditionalGeneration = None
AutoConfig = None
AutoModelForImageTextToText = None
process_vision_info = None
PeftModel = None
HAS_VLM_AUTO = True

try:
    import spaces  # Hugging Face ZeroGPU helper (pip install spaces)
    HAS_SPACES = True
except Exception:
    HAS_SPACES = False

    class _DummySpaces:
        @staticmethod
        def GPU(fn=None, **kwargs):
            def wrap(f):
                return f
            return wrap(fn) if callable(fn) else wrap

    spaces = _DummySpaces()

# ── Gradio 6 + Spaces: neutralise Gradio's automatic GPU auto-wrap ───────────
# Gradio 6.27 calls spaces.gradio_auto_wrap() on every event handler, which
# would wrap `transcribe` in a second @spaces.GPU (nested GPU calls are not
# supported by ZeroGPU) — and on some Spaces that attribute is missing
# altogether, killing the app at startup with:
#   AttributeError: module 'spaces' has no attribute 'gradio_auto_wrap'.
# We already decorate the Qwen call with @spaces.GPU explicitly, so switch the
# automatic wrapping off and provide a no-op if the symbol is absent.
if HAS_SPACES:
    try:
        if hasattr(spaces, "disable_gradio_auto_wrap"):
            spaces.disable_gradio_auto_wrap()
        if not hasattr(spaces, "gradio_auto_wrap"):
            spaces.gradio_auto_wrap = lambda task, *a, **k: task
    except Exception as _e:
        print(f"[eym-space] spaces auto-wrap shim skipped: {_e}", flush=True)

# ── Config ────────────────────────────────────────────────────────────────────
QWEN_ID = os.environ.get("EYM_QWEN_MODEL_ID",
                         "diabolic6045/Sanskrit-Qwen2.5-VL-7B-Instruct-OCR")
TROCR_ID = os.environ.get("EYM_TROCR_MODEL_ID", "paudelanil/trocr-devanagari-2")
TROCR_ADAPTER = os.environ.get("EYM_TROCR_ADAPTER_ID", "").strip()
SKIP_QWEN = os.environ.get("EYM_SPACE_NO_QWEN", "0") == "1"
# The vision engine is model-agnostic: point it at any supported
# image-text-to-text checkpoint. Defaults stay on the Sanskrit Qwen fine-tune.
#   Qwen2.5-VL family : Qwen/Qwen2.5-VL-3B-Instruct, snskrt/qwen2-5-vl-sanskrit-ocr
#   Qwen3-VL family   : Qwen/Qwen3-VL-2B-Instruct, Qwen/Qwen3-VL-4B-Instruct
#   Qwen2-VL family   : snskrt/sanskrit-ocr-qwen2vl
#   Gemma 3 family    : snskrt/gemma-3-4b-it-sanskrit-ocr
VISION_ID = (os.environ.get("EYM_VISION_MODEL_ID", "").strip()
             or os.environ.get("EYM_QWEN_MODEL_ID", "").strip() or QWEN_ID)
VISION_MAX_TOKENS = int(os.environ.get("EYM_QWEN_MAX_TOKENS", "512"))
VISION_MAX_SIDE = int(os.environ.get("EYM_QWEN_MAX_SIDE", "1536"))
VISION_4BIT = os.environ.get("EYM_VISION_4BIT", "1") != "0"
TROCR_BEAMS = int(os.environ.get("EYM_TROCR_BEAMS", "4"))
TROCR_MAX_LEN = int(os.environ.get("EYM_TROCR_MAX_LEN", "64"))
TROCR_BATCH = max(1, int(os.environ.get("EYM_TROCR_BATCH", "4")))
WORD_GAP = float(os.environ.get("EYM_WORD_GAP_FACTOR", "0.30"))

# ── MIDF / Kraken PP-OCRv6 (tadad/midf-sanskrit-ocr) ──────────────────────────
# 17 MB of weights (12 MB recogniser + 5 MB line segmenter), CPU-only, Apache-2.0.
# Measured here on 17 rendered Devanagari crops (mean character error rate):
#   line crops, ink-trimmed ............ 0.053   (TrOCR on the same set: 0.884)
#   word crops, ink-trimmed ............ 0.194   (TrOCR: 0.611)
#   untrimmed crops .................... 0.146
#   forced to a squat 2:1 shape ........ 1.000   (returns empty text)
# So: always ink-trim, never pad the crop outwards, and feed whole lines.
MIDF_ID = os.environ.get("EYM_MIDF_MODEL_ID", "tadad/midf-sanskrit-ocr")
MIDF_REC_FILE = os.environ.get("EYM_MIDF_REC_FILE", "recognition/model.safetensors")
MIDF_SEG_FILE = os.environ.get("EYM_MIDF_SEG_FILE", "segmentation/model.safetensors")
MIDF_MODE = os.environ.get("EYM_MIDF_MODE", "page").strip().lower()   # page (default: MIDF's own trained segmenter) | line
MIDF_ON = os.environ.get("EYM_MIDF", "1") != "0"
MIDF_TRIM_PAD = int(os.environ.get("EYM_MIDF_TRIM_PAD", "6"))
MIDF_MAX_H = int(os.environ.get("EYM_MIDF_MAX_H", "160"))   # 0 = never rescale
MIDF_WORD_GAP = float(os.environ.get("EYM_MIDF_WORD_GAP", "0.14"))
# MIDF emits no spaces. With this on, the line text is rebuilt by joining the
# detected word slices with spaces (set 0 to keep the raw space-less string).
MIDF_SPACES = os.environ.get("EYM_MIDF_SPACES", "1") != "0"
# kraken is a heavy import (lightning/torchvision); detect it cheaply instead.
HAS_KRAKEN = importlib.util.find_spec("kraken") is not None

QWEN_PROMPT = ("Please transcribe the Sanskrit text shown in this image. "
               "Output only the transcribed Devanagari text, with no commentary, "
               "translation, or extra formatting.")
VISION_PROMPT = os.environ.get("EYM_VISION_PROMPT", "").strip() or QWEN_PROMPT

_trocr_model = None
_trocr_proc = None
_vision_model = None
_vision_proc = None
_vision_ready = False
_vision_error = None
_vision_mtype = None      # resolved model_type, e.g. qwen2_5_vl / gemma3
_midf_rec = None          # Kraken PP-OCRv6 recogniser
_midf_seg = None          # Kraken BLLA segmenter (only in MIDF_MODE=page)
_midf_ready = False
_midf_error = None
_lock = threading.Lock()


def _log(msg):
    print(f"[eym-space] {msg}", flush=True)


# ── transformers 4.x / 5.x compatibility ──────────────────────────────────────
# Hugging Face Spaces always installs its own Gradio (6.x), which requires
# huggingface-hub>=1.16 — so transformers 5.x is what lands on the Space.
# Keep both generations working: 5.x drops `torch_dtype` (renamed `dtype`) and
# can no longer auto-resolve this TrOCR checkpoint's processor.
def _tf_major():
    try:
        import transformers as _tf
        return int(str(_tf.__version__).split(".")[0])
    except Exception:
        return 4


TF_MAJOR = _tf_major()


def _build_trocr_processor():
    """TrOCRProcessor that works on transformers 4.x and 5.x.

    The checkpoint has no tokenizer.json (fast) file and its
    preprocessor_config.json points at the removed ViTFeatureExtractor, so on
    transformers 5 the auto classes cannot resolve it. Build it explicitly.
    """
    try:
        from transformers import TrOCRProcessor
        return TrOCRProcessor.from_pretrained(TROCR_ID)
    except Exception as e:
        _log(f"TrOCRProcessor.from_pretrained failed ({e}) — building explicitly")
    from transformers import ViTImageProcessor, TrOCRProcessor
    ip = ViTImageProcessor.from_pretrained(TROCR_ID)
    tok = None
    try:
        from transformers import RobertaTokenizer  # slow; still present in v5
        tok = RobertaTokenizer.from_pretrained(TROCR_ID)
    except Exception as e:
        _log(f"RobertaTokenizer failed ({e}) — trying AutoTokenizer")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TROCR_ID)
    return TrOCRProcessor(image_processor=ip, tokenizer=tok)


# ═══════════════════════════════════════════════════════════════════════════════
# Segmentation (OpenCV v2 — same pipeline as ocr_server.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _erase_frame_lines(mask):
    h, w = mask.shape
    ink = mask > 0
    row_frac = ink.sum(axis=1) / float(w)
    col_frac = ink.sum(axis=0) / float(h)
    frame_rows = row_frac > 0.70
    frame_cols = col_frac > 0.70
    top_lim, bot_lim = int(h * 0.12), h - int(h * 0.12)
    left_lim, right_lim = int(w * 0.12), w - int(w * 0.12)

    def _edge_runs(flags, lo_guard, hi_guard, n):
        runs, start = [], None
        for i, v in enumerate(flags):
            if v and start is None:
                start = i
            elif not v and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, n))
        return [(a, b) for a, b in runs if a < lo_guard or b > hi_guard]

    cleaned = mask.copy()
    for a, b in _edge_runs(frame_rows, top_lim, bot_lim, h):
        cleaned[a:b, :] = 0
    for a, b in _edge_runs(frame_cols, left_lim, right_lim, w):
        cleaned[:, a:b] = 0
    return cleaned


def _remove_scan_borders(mask, img_w, img_h):
    cleaned = _erase_frame_lines(mask)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        touches = (x <= 2 or y <= 2 or x + w >= img_w - 2 or y + h >= img_h - 2)
        if touches and (w > img_w * 0.5 or h > img_h * 0.5
                        or cv2.contourArea(c) > img_w * img_h * 0.02):
            cv2.drawContours(cleaned, [c], -1, 0, -1)
    return cleaned


def _merge_overlapping(boxes):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[1])
    merged = [boxes[0]]
    for x, y, w, h in boxes[1:]:
        px, py, pw, ph = merged[-1]
        if min(y + h, py + ph) - max(y, py) > 0.5 * min(h, ph):
            nx0, ny0 = min(px, x), min(py, y)
            nx1, ny1 = max(px + pw, x + w), max(py + ph, y + h)
            merged[-1] = (nx0, ny0, nx1 - nx0, ny1 - ny0)
        else:
            merged.append((x, y, w, h))
    return merged


def segment_lines(gray):
    """numpy grayscale → list of {bbox_norm, crop_gray}."""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 51, 9)
    mask = _remove_scan_borders(mask, w, h)
    # 2x2 (was 3x3): a 3x3 opening erased thin hyphens and dashes outright
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    join_w = max(20, w // 40)
    dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (join_w, 3)))
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    allb = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in allb if 10 <= b[3] <= h * 0.25 and b[2] >= w * 0.05]
    boxes = _merge_overlapping(boxes)
    if boxes:
        med = float(np.median([b[3] for b in boxes]))
        small = [b for b in allb if b not in boxes]
        boxes = [b for b in boxes if b[3] >= max(10.0, 0.4 * med)]
        # Small marks (a dash —, hyphen, asterisk, daṇḍa, numeral) used to be
        # thrown away by the size filter. Attach each one to the line it sits
        # on by widening that line's box, so the recogniser actually sees it.
        boxes = [list(b) for b in boxes]
        for sx, sy, sw, sh in small:
            if sw * sh < 4:
                continue
            cy = sy + sh / 2.0
            for b in boxes:
                if b[1] - 0.25 * b[3] <= cy <= b[1] + 1.25 * b[3]:
                    x0, x1 = min(b[0], sx), max(b[0] + b[2], sx + sw)
                    b[0], b[2] = x0, x1 - x0
                    break
        boxes = [tuple(b) for b in boxes]
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    if not boxes:
        boxes = [(0, 0, w, h)]
    out, pad = [], 4
    for x, y, bw, bh in boxes:
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        out.append({"bbox": {"xmin": x0 / w, "ymin": y0 / h,
                             "xmax": x1 / w, "ymax": y1 / h},
                    "crop": gray[y0:y1, x0:x1]})
    return out


def split_words(strip_gray, gap_factor=None):
    """One line strip → list of {x0, x1, crop_gray} word crops.

    gap_factor=None keeps the classic behaviour (gap scales with the strip
    height, which is right when the strip is one tight line of a page). The
    MIDF path passes EYM_MIDF_WORD_GAP instead: when someone uploads a single
    line the "strip" is the whole image, whose height is far larger than the
    glyphs, and every word would merge into one.
    """
    lh, lw = strip_gray.shape
    _, mask = cv2.threshold(strip_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = np.sum(mask, axis=0) > (lh * 2)
    runs, start = [], None
    for x in range(lw):
        if ink[x] and start is None:
            start = x
        elif not ink[x] and start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, lw))
    if not runs:
        return [{"x0": 0, "x1": lw, "crop": strip_gray}]
    if gap_factor is None:
        min_gap = max(6, int(lh * WORD_GAP))
    else:
        rows = np.where(mask.any(axis=1))[0]
        ink_h = int(rows[-1] - rows[0] + 1) if len(rows) else lh
        min_gap = max(4, int(ink_h * gap_factor))
    words = [runs[0]]
    for s, e in runs[1:]:
        ps, pe = words[-1]
        words[-1] = (ps, e) if s - pe < min_gap else words[-1]
        if s - pe >= min_gap:
            words.append((s, e))
    if len(words) > 1:
        words = [(s, e) for s, e in words if e - s >= 5] or words
    out, pad = [], 3
    for s, e in words:
        s0, e0 = max(0, s - pad), min(lw, e + pad)
        out.append({"x0": s0, "x1": e0, "crop": strip_gray[:, s0:e0]})
    return out


def _word_to_pil(crop_gray):
    """DevGen-style word normalisation → 224×224 PIL RGB."""
    if crop_gray.ndim == 2:
        bgr = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    else:
        bgr = crop_gray
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.dilate(mask, k, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    boxes = [cv2.boundingRect(c) for c in contours
             if cv2.contourArea(c) >= max(12, int(h * w * 0.0001))]
    if boxes:
        x1 = min(x for x, _, _, _ in boxes)
        y1 = min(y for _, y, _, _ in boxes)
        x2 = max(x + bw for x, _, bw, _ in boxes)
        y2 = max(y + bh for _, y, _, bh in boxes)
        px = max(8, int((x2 - x1) * 0.18))
        py = max(8, int((y2 - y1) * 0.18))
        bgr = bgr[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
    hh, ww = bgr.shape[:2]
    sc = min(224 / max(hh, 1), 224 / max(ww, 1))
    nh, nw = max(1, int(hh * sc)), max(1, int(ww * sc))
    rs = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.ones((224, 224, 3), dtype=np.uint8) * 255
    yo, xo = (224 - nh) // 2, (224 - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = rs
    return PILImage.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def analyze_medium(pil_img):
    """Lightweight medium/degradation heuristic (no lexicon/SQLite on Space)."""
    try:
        img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        mean_v, std_v = float(np.mean(gray)), float(np.std(gray))
        deg = round(min(1.0, max(0.0, (128.0 - std_v) / 128.0)), 2)
        cls = "degraded_paper" if (mean_v < 150 or std_v > 65) else "paper_ink"
        return {"medium_class": cls, "confidence": 0.85,
                "degradation_score": deg,
                "details": {"mean_luminance": round(mean_v, 1),
                            "contrast_std": round(std_v, 1)}}
    except Exception:
        return {"medium_class": "paper_ink", "confidence": 0.75,
                "degradation_score": 0.2}


# ═══════════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════════

def load_trocr():
    global _trocr_model, _trocr_proc
    if _trocr_model is not None:
        return True
    with _lock:
        if _trocr_model is not None:
            return True
        from transformers import VisionEncoderDecoderModel
        _log(f"loading TrOCR {TROCR_ID} …")
        proc = _build_trocr_processor()
        model = VisionEncoderDecoderModel.from_pretrained(TROCR_ID)
        tok = proc.tokenizer
        if getattr(tok, "cls_token_id", None) is not None:
            model.config.decoder_start_token_id = tok.cls_token_id
        if getattr(tok, "pad_token_id", None) is not None:
            model.config.pad_token_id = tok.pad_token_id
        if getattr(tok, "sep_token_id", None) is not None:
            model.config.eos_token_id = tok.sep_token_id
        try:
            model.config.vocab_size = model.config.decoder.vocab_size
        except Exception:
            pass
        if TROCR_ADAPTER and HAS_PEFT:
            try:
                _log(f"loading TrOCR adapter {TROCR_ADAPTER} …")
                pm = PeftModel.from_pretrained(model, TROCR_ADAPTER)
                try:
                    model = pm.merge_and_unload()
                except Exception:
                    model = pm
            except Exception as e:
                _log(f"adapter failed ({e}) — base model continues")
        # ZeroGPU: TrOCR runs OUTSIDE the @spaces.GPU function, so moving it to
        # "cuda" here gives it placeholder weights and it emits the same junk
        # ("$%&'()*+,-./0123…") for every image. Keep it on CPU.
        device = "cpu"
        model = model.to(device).eval()
        _trocr_model, _trocr_proc = model, proc
        _log(f"TrOCR ready on {device} ✓")
        return True


def trocr_decode(pil_list):
    px = _trocr_proc(images=pil_list, return_tensors="pt").pixel_values
    px = px.to(_trocr_model.device)
    gen = dict(num_beams=TROCR_BEAMS, max_length=TROCR_MAX_LEN,
               early_stopping=True, return_dict_in_generate=True,
               output_scores=True)
    try:
        gen["decoder_start_token_id"] = _trocr_model.config.decoder_start_token_id
    except Exception:
        pass
    with torch.no_grad():
        out = _trocr_model.generate(px, **gen)
    texts = _trocr_proc.batch_decode(out.sequences, skip_special_tokens=True)
    confs = [None] * len(texts)
    try:
        bi = getattr(out, "beam_indices", None)
        kw = dict(normalize_logits=True)
        if bi is not None:
            kw["beam_indices"] = bi
        scores = _trocr_model.compute_transition_scores(out.sequences, out.scores, **kw)
        probs = torch.exp(scores)
        confs = [float(probs[i].mean().item()) for i in range(len(texts))]
    except Exception as e:
        _log(f"trocr confidence skipped: {e}")
    return [t.strip() for t in texts], confs


def _vision_compute_dtype():
    # T4 (ZeroGPU) has no bfloat16 — use float16 there, bf16 on Ampere+.
    try:
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def _resolve_vision_type(model_id):
    """Read the repo's model_type once (config only — a few KB)."""
    if AutoConfig is not None:
        try:
            return AutoConfig.from_pretrained(model_id,
                                              trust_remote_code=True).model_type
        except Exception as e:
            _log(f"AutoConfig failed ({e}) — reading config.json instead")
    try:
        import json as _json
        from huggingface_hub import hf_hub_download
        return _json.load(open(hf_hub_download(model_id, "config.json"),
                               encoding="utf-8")).get("model_type")
    except Exception as e:
        _log(f"config.json read failed ({e})")
        return None


def _vision_family(mtype):
    m = (mtype or "").lower()
    if m.startswith("qwen"):
        return "qwen"
    if m.startswith("gemma"):
        return "gemma"
    return "other"


def _vision_engine_name(mtype):
    # 'qwen-vl' keeps the label the site already renders; every other family is
    # reported under its own model_type (gemma3, …) so you can see what answered.
    return "qwen-vl" if _vision_family(mtype) == "qwen" else (mtype or "vision")


_IMAGE_MARKERS = {
    "gemma": ("<start_of_image>",),
    "qwen": ("<|vision_start|>", "<|image_pad|>"),
    "other": (),
}


def _has_image_placeholder(text, family):
    return any(m in (text or "") for m in _IMAGE_MARKERS.get(family, ()))


def _fallback_prompt(prompt, family):
    """Raw chat text for checkpoints whose repo ships no chat template.

    Without this the image marker is silently dropped and generation dies with
    'Image features and image tokens do not match'.
    """
    if family == "gemma":
        return ("<bos><start_of_turn>user\n<start_of_image>" + prompt
                + "<end_of_turn>\n<start_of_turn>model\n")
    if family == "qwen":
        return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                + prompt + "<|im_end|>\n<|im_start|>assistant\n")
    return prompt


def _vision_build_inputs(proc, pil_img, prompt, family):
    """Chat template + pixel inputs, built per model family."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_img},
        {"type": "text", "text": prompt}]}]
    text = None
    try:
        text = proc.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True)
    except Exception as e:
        _log(f"apply_chat_template failed ({e}) — using the raw prompt form")
    if not _has_image_placeholder(text, family):
        if _IMAGE_MARKERS.get(family):
            _log("no image placeholder in the rendered prompt — "
                 "falling back to the raw chat form")
        text = _fallback_prompt(prompt, family)
    if family == "qwen" and process_vision_info is not None:
        try:
            ii, vi = process_vision_info(messages)
            return proc(text=[text], images=ii, videos=vi, padding=True,
                        return_tensors="pt")
        except Exception as e:
            _log(f"process_vision_info failed ({e}) — passing the image directly")
    # Gemma 3 and anything else: the processor consumes the PIL image directly.
    return proc(text=[text], images=[pil_img], padding=True, return_tensors="pt")


def gpu_decorator(duration=120):
    """Lazy stand-in for @spaces.GPU.

    Resolving `spaces.GPU` at decoration time breaks when the early
    `import spaces` fails (which is exactly what the missing-symbol crash on
    Spaces looks like from the inside) — the decorator would silently become a
    no-op and the vision model would never get a GPU. Bind it on first call
    instead, by which point every module is loaded.
    """
    def decorate(fn):
        bound = None

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal bound
            if bound is None:
                try:
                    import spaces as _sp
                    if hasattr(_sp, "GPU"):
                        bound = _sp.GPU(duration=duration)(fn)
                    else:
                        bound = fn
                except Exception as e:   # no ZeroGPU here — run on CPU
                    _log(f"spaces.GPU unavailable ({e}) — running on CPU")
                    bound = fn
            return bound(*args, **kwargs)
        return wrapper
    return decorate


@gpu_decorator(duration=120)
def _vision_transcribe_gpu(pil_img):
    """Vision-model load (once, cached on the worker) + inference. MUST stay
    inside the @spaces.GPU function — ZeroGPU only provides CUDA there."""
    global _vision_model, _vision_proc, _vision_ready, _vision_error, _vision_mtype
    with _lock:
        if _vision_model is None:
            _log(f"loading vision model {VISION_ID} …")
            from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
            if _vision_mtype is None:
                _vision_mtype = _resolve_vision_type(VISION_ID)
            _log(f"resolved model_type: {_vision_mtype}")
            cdtype = _vision_compute_dtype()
            _vision_proc = AutoProcessor.from_pretrained(VISION_ID,
                                                         trust_remote_code=True)
            kw = dict(trust_remote_code=True, device_map="auto")
            if VISION_4BIT and torch.cuda.is_available():
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=cdtype)
            # transformers 5 renamed torch_dtype -> dtype
            kw["dtype" if TF_MAJOR >= 5 else "torch_dtype"] = cdtype
            _vision_model = AutoModelForImageTextToText.from_pretrained(VISION_ID, **kw)
            _vision_model.eval()
            _vision_ready = True
            _log(f"vision model ready ✓ ({type(_vision_model).__name__})")
    w, h = pil_img.size
    if max(w, h) > VISION_MAX_SIDE:
        sc = VISION_MAX_SIDE / float(max(w, h))
        pil_img = pil_img.resize((max(1, int(w * sc)), max(1, int(h * sc))),
                                 PILImage.LANCZOS)
    inputs = _vision_build_inputs(_vision_proc, pil_img, VISION_PROMPT,
                                  _vision_family(_vision_mtype))
    try:
        dev = next(_vision_model.parameters()).device
    except Exception:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    gen = dict(max_new_tokens=VISION_MAX_TOKENS, do_sample=False, use_cache=True,
               repetition_penalty=1.1)
    try:
        gen["pad_token_id"] = _vision_proc.tokenizer.eos_token_id
    except Exception:
        pass
    t0 = time.time()
    with torch.no_grad():
        generated = _vision_model.generate(**inputs, **gen)
    out = generated[:, inputs["input_ids"].shape[1]:]
    txt = _vision_proc.batch_decode(out, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)[0].strip()
    _log(f"vision model decoded {len(txt)} chars in {time.time()-t0:.1f}s")
    return txt


def vision_available():
    if SKIP_QWEN or not (HAS_VLM_AUTO or HAS_QWEN_CLS):
        return False
    # A multi-GB vision model on CPU would OOM/crawl — GPU-only by design.
    return torch.cuda.is_available() or HAS_SPACES


# ═══════════════════════════════════════════════════════════════════════════════
# EasyOCR (Devanagari: hi, mr, ne) — Whole-page OCR + Bounding Boxes
# ═══════════════════════════════════════════════════════════════════════════════

_easyocr_reader = None
_easyocr_ready = False
_easyocr_error = None

EASYOCR_ON = os.environ.get("EYM_EASYOCR", "1") != "0"


def easyocr_available():
    """True when easyocr is installed and enabled."""
    if not EASYOCR_ON:
        return False
    try:
        import easyocr  # noqa: F401
        return True
    except Exception:
        return False


def load_easyocr():
    """Load EasyOCR Devanagari models (hi, mr, ne) on CPU/GPU."""
    global _easyocr_reader, _easyocr_ready, _easyocr_error
    if not easyocr_available() or _easyocr_ready or _easyocr_error is not None:
        return _easyocr_ready
    with _lock:
        if _easyocr_ready or _easyocr_error is not None:
            return _easyocr_ready
        try:
            import easyocr
            _log("loading EasyOCR (Devanagari: hi, mr, ne) …")
            # ZeroGPU: calling torch.cuda.* outside an @spaces.GPU function
            # aborts with "Low-level CUDA init reached". EasyOCR is fast enough
            # on CPU, so never ask for the GPU here.
            use_gpu = False
            _easyocr_reader = easyocr.Reader(["hi", "mr", "ne"], gpu=use_gpu,
                                             quantize=False)
            _easyocr_ready = True
            _log(f"EasyOCR ready on {'cuda' if use_gpu else 'cpu'} ✓")
        except Exception as e:
            _easyocr_error = f"{type(e).__name__}: {e}"
            _log(f"EasyOCR unavailable ({_easyocr_error})")
    return _easyocr_ready


def easyocr_transcribe_image(pil_img):
    """Run EasyOCR over full page or line crop, returning structured lines + boxes."""
    if not load_easyocr():
        raise RuntimeError(f"EasyOCR failed to initialize: {_easyocr_error}")

    w_img, h_img = pil_img.size
    np_img = np.array(pil_img.convert("L")) if isinstance(pil_img, PILImage.Image) else np.array(pil_img)
    raw = _easyocr_reader.readtext(np_img, detail=1, batch_size=1)

    if not raw:
        return {
            "text": "",
            "lines": [],
            "engine": "easyocr-devanagari",
            "medium": "cloud",
            "note": "No text detected by EasyOCR."
        }

    raw_lines = []
    for item in raw:
        bbox, txt, conf = item[0], item[1].strip(), float(item[2])
        if not txt:
            continue
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - min(xs)), int(max(ys) - min(ys))
        raw_lines.append({
            "box": [x, y, w, h],
            "text": txt,
            "confidence": round(conf, 3),
            "y_center": y + h / 2.0,
            "x_left": x
        })

    raw_lines.sort(key=lambda l: (l["box"][1], l["box"][0]))

    merged_lines = []
    for line in raw_lines:
        placed = False
        for m in merged_lines:
            m_y_center = m["box"][1] + m["box"][3] / 2.0
            line_y_center = line["y_center"]
            avg_h = (m["box"][3] + line["box"][3]) / 2.0
            if abs(m_y_center - line_y_center) < avg_h * 0.5:
                new_x = min(m["box"][0], line["box"][0])
                new_y = min(m["box"][1], line["box"][1])
                new_r = max(m["box"][0] + m["box"][2], line["box"][0] + line["box"][2])
                new_b = max(m["box"][1] + m["box"][3], line["box"][1] + line["box"][3])
                m["box"] = [new_x, new_y, new_r - new_x, new_b - new_y]
                if line["x_left"] > m.get("last_x", m["box"][0]):
                    m["text"] = m["text"] + " " + line["text"]
                else:
                    m["text"] = line["text"] + " " + m["text"]
                m["confidence"] = round((m["confidence"] + line["confidence"]) / 2.0, 3)
                m["last_x"] = line["x_left"]
                placed = True
                break
        if not placed:
            line_copy = dict(line)
            line_copy["last_x"] = line["x_left"]
            merged_lines.append(line_copy)

    lines_out = []
    full_texts = []
    for ml in merged_lines:
        txt = ml["text"].strip()
        full_texts.append(txt)
        words = txt.split()
        box = ml["box"]
        norm_bbox = {
            "xmin": max(0.0, min(1.0, box[0] / float(max(1, w_img)))),
            "ymin": max(0.0, min(1.0, box[1] / float(max(1, h_img)))),
            "xmax": max(0.0, min(1.0, (box[0] + box[2]) / float(max(1, w_img)))),
            "ymax": max(0.0, min(1.0, (box[1] + box[3]) / float(max(1, h_img)))),
        }
        w_objs = []
        if len(words) > 0:
            char_total = sum(len(w) for w in words)
            cur_x = box[0]
            for w in words:
                w_width = int(box[2] * (len(w) / max(1, char_total)))
                w_norm = {
                    "xmin": max(0.0, min(1.0, cur_x / float(max(1, w_img)))),
                    "ymin": norm_bbox["ymin"],
                    "xmax": max(0.0, min(1.0, (cur_x + w_width) / float(max(1, w_img)))),
                    "ymax": norm_bbox["ymax"],
                }
                w_objs.append({
                    "text": w,
                    "bbox": w_norm,
                    "box": [cur_x, box[1], w_width, box[3]],
                    "confidence": ml["confidence"],
                    "approx": True
                })
                cur_x += w_width
        lines_out.append({
            "bbox": norm_bbox,
            "box": box,
            "text": txt,
            "confidence": ml["confidence"],
            "words": w_objs,
            "granularity": "line"
        })

    return {
        "text": "\n".join(full_texts),
        "lines": lines_out,
        "engine": "easyocr-devanagari",
        "medium": "cloud",
        "note": "Extracted with EasyOCR Devanagari engine."
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MIDF (tadad/midf-sanskrit-ocr) — Kraken BLLA + PP-OCRv6, CPU, 17 MB
# ═══════════════════════════════════════════════════════════════════════════════

def midf_available():
    """True when kraken is installed and the engine has not been switched off."""
    return bool(MIDF_ON and HAS_KRAKEN)


def load_midf():
    """Fetch the 17 MB of Kraken weights from the Hub (idempotent, thread-safe)."""
    global _midf_rec, _midf_seg, _midf_ready, _midf_error
    if not midf_available() or _midf_ready or _midf_error is not None:
        return _midf_ready
    with _lock:
        if _midf_ready or _midf_error is not None:
            return _midf_ready
        try:
            from huggingface_hub import hf_hub_download
            from kraken.tasks import RecognitionTaskModel, SegmentationTaskModel
            _log(f"loading MIDF {MIDF_ID} …")
            rec_path = hf_hub_download(MIDF_ID, MIDF_REC_FILE)
            _midf_rec = RecognitionTaskModel.load_model(rec_path)
            if MIDF_MODE == "page":
                seg_path = hf_hub_download(MIDF_ID, MIDF_SEG_FILE)
                _midf_seg = SegmentationTaskModel.load_model(seg_path)
            _midf_ready = True
            _log(f"MIDF (Kraken PP-OCRv6) ready on cpu ✓ "
                 f"[mode={MIDF_MODE}]")
        except Exception as e:
            _midf_error = f"{type(e).__name__}: {e}"
            _log(f"MIDF unavailable ({_midf_error}) — TrOCR will cover")
    return _midf_ready


def _midf_prepare(pil):
    """Ink-trim a crop; never pad it outwards.

    Measured CER on 17 rendered Devanagari crops: untrimmed 0.146, trimmed
    0.094, trimmed + rescaled 0.050. Padding a crop into a squat shape makes
    PP-OCRv6 return empty text (CER 1.000), so the aspect ratio is left alone.
    """
    if not CV2_AVAILABLE:
        return pil
    try:
        gray = np.array(pil.convert("L"))
        ink = gray < 245
        if not ink.any():
            return pil
        rows = np.where(ink.any(1))[0]
        cols = np.where(ink.any(0))[0]
        p = max(0, MIDF_TRIM_PAD)
        im = pil.crop((max(0, cols[0] - p), max(0, rows[0] - p),
                       min(pil.width, cols[-1] + 1 + p),
                       min(pil.height, rows[-1] + 1 + p)))
        if MIDF_MAX_H and im.height > MIDF_MAX_H:
            w = max(8, int(round(im.width * MIDF_MAX_H / im.height)))
            im = im.resize((w, MIDF_MAX_H), PILImage.LANCZOS)
        return im
    except Exception as e:
        _log(f"midf prepare skipped ({e})")
        return pil


def _midf_bounds(im):
    """Wrap a single image as a one-line Kraken segmentation."""
    from kraken.containers import BaselineLine, Segmentation as KrakenSeg
    w, h = im.size
    return KrakenSeg(
        type="baselines", imagename="crop", text_direction="horizontal-lr",
        script_detection=False,
        lines=[BaselineLine(id="l0", baseline=[(0, h - 1), (w - 1, h - 1)],
                            boundary=[(0, 0), (w - 1, 0), (w - 1, h - 1),
                                      (0, h - 1)], text="")],
        regions={}, line_orders=[], language=None)


def midf_recognize_lines(pils):
    """Recognise each PIL crop as one text line → [(text, mean_confidence)]."""
    if not load_midf():
        raise RuntimeError("MIDF not available")
    from kraken.configs import RecognitionInferenceConfig
    cfg = RecognitionInferenceConfig(accelerator="cpu", device=1,
                                     bidi_reordering=False, num_line_workers=0)
    out = []
    for im in pils:
        recs = list(_midf_rec.predict(im, _midf_bounds(im), cfg))
        rec = recs[0] if recs else None
        text = (rec.prediction or "") if rec else ""
        conf = (sum(rec.confidences) / len(rec.confidences)
                if rec is not None and rec.confidences else None)
        out.append((text, conf))
    return out


def midf_page_lines(pil):
    """Full Kraken page pipeline (needs ~2 GB RAM) → [(text, bbox_norm, conf)].

    Only used when EYM_MIDF_MODE=page. Line bbox comes from the Kraken polygon.
    """
    if not load_midf() or _midf_seg is None:
        raise RuntimeError("MIDF page mode unavailable")
    from dataclasses import replace
    from kraken.configs import (RecognitionInferenceConfig,
                                SegmentationInferenceConfig)
    from kraken.lib.segmentation import calculate_polygonal_environment

    common = {"accelerator": "cpu", "device": 1}
    bounds = _midf_seg.predict(pil, SegmentationInferenceConfig(**common))
    if not bounds.lines:
        return []
    polys = calculate_polygonal_environment(
        pil, [l.baseline for l in bounds.lines], scale=(1000, 0), topline=True)
    bounds = replace(bounds, lines=[
        replace(l, boundary=p.tolist() if hasattr(p, "tolist") else (p or l.boundary))
        for l, p in zip(bounds.lines, polys)])
    recs = list(_midf_rec.predict(
        pil, bounds,
        RecognitionInferenceConfig(**common, bidi_reordering=False,
                                   num_line_workers=0)))
    W, H = pil.size
    out = []
    for line, rec in zip(bounds.lines, recs):
        poly = line.boundary or []
        xs = [pt[0] for pt in poly] or [0]
        ys = [pt[1] for pt in poly] or [0]
        out.append({
            "text": rec.prediction or "",
            "confidence": (sum(rec.confidences) / len(rec.confidences)
                           if rec.confidences else None),
            "bbox": {"xmin": max(0.0, min(xs) / W), "ymin": max(0.0, min(ys) / H),
                     "xmax": min(1.0, max(xs) / W), "ymax": min(1.0, max(ys) / H)},
        })
    return out


def _grapheme_safe(text, idx):
    """Nudge a cut point right so it never splits before a combining mark.

    Devanagari vowel signs and virama are Mn/Mc; cutting in front of one would
    paste the sign onto the previous word.
    """
    while idx < len(text) and (unicodedata.combining(text[idx])
                               or unicodedata.category(text[idx]) in ("Mn", "Mc", "Me")):
        idx += 1
    return idx


def apportion_words(text, spans):
    """Split a space-less line across detected word spans by relative width.

    MIDF emits no spaces, so word boxes stay approximate (the line text itself
    is exact). The UI edits word by word, so we hand it per-word slices sized
    by how wide each detected word gap is.
    """
    n = len(spans)
    if n <= 1 or not text:
        return [text] if text else [""] * max(n, 1)
    widths = [max(1.0, s[1] - s[0]) for s in spans]
    total = float(sum(widths))
    counts, used = [], 0
    for i, wd in enumerate(widths):
        if i == n - 1:
            counts.append(max(0, len(text) - used))
            break
        c = int(round(len(text) * wd / total))
        c = max(1, min(c, len(text) - used - (n - i - 1)))
        counts.append(c)
        used += c
    parts, pos = [], 0
    for c in counts:
        end = _grapheme_safe(text, pos + c)
        parts.append(text[pos:end])
        pos = end
    if pos < len(text) and parts:
        parts[-1] += text[pos:]
    return parts


# ═══════════════════════════════════════════════════════════════════════════════
# AI Sanskrit Post-Correction & Mistake Analyzer (Mistral / AI4Bharat / DeepSeek)
# ═══════════════════════════════════════════════════════════════════════════════

AI_CORRECT_MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "ai4bharat": "ai4bharat/airavata",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "sarvam": "sarvamai/sarvam-2b"
}
_correct_model = None
_correct_tokenizer = None
_loaded_model_id = None


def _resolve_ai_model_id(choice):
    key = str(choice or "mistral").lower().strip()
    return AI_CORRECT_MODELS.get(key, AI_CORRECT_MODELS["mistral"])


@gpu_decorator(duration=60)
def _run_ai_correction_gpu(noisy_text, model_id):
    global _correct_model, _correct_tokenizer, _loaded_model_id
    from transformers import AutoModelForCausalLM, AutoTokenizer
    with _lock:
        if _correct_model is None or _loaded_model_id != model_id:
            _log(f"loading AI Sanskrit model {model_id} …")
            _correct_tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True)
            kw = dict(trust_remote_code=True, device_map="auto")
            if torch.cuda.is_available():
                kw["torch_dtype"] = (torch.bfloat16
                                     if torch.cuda.get_device_capability()[0] >= 8
                                     else torch.float16)
            _correct_model = AutoModelForCausalLM.from_pretrained(
                model_id, **kw).eval()
            _loaded_model_id = model_id
            _log(f"AI Sanskrit model ready on {_correct_model.device} ✓ ({model_id})")

    prompt = f"""You are an expert Classical Sanskrit grammarian, philologist, and manuscript editor.
Analyze this noisy OCR transcription of Sanskrit text:
\"{noisy_text}\"

Instructions:
1. Provide the complete grammatically and orthographically corrected Sanskrit text in Devanagari script. Fix broken conjuncts (saṁyuktākṣaras), missing virāmas/mātrās, and sandhi boundary splits.
2. Itemize every specific mistake you found, showing the original error, the corrected form, and a concise explanation in English.

Output strictly as a valid JSON object matching this schema:
{{
  "corrected_text": "<full corrected Sanskrit text in Devanagari>",
  "corrections": [
    {{"original": "<mistaken word/phrase>", "corrected": "<corrected word>", "reason": "<explanation in English>"}}
  ],
  "notes": "<meter / grammar observation>"
}}
Return ONLY raw JSON without markdown code fences."""

    messages = [
        {"role": "system", "content": "You are a precise Sanskrit linguistic scholar. Output valid JSON only."},
        {"role": "user", "content": prompt}
    ]
    text = _correct_tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
    inputs = _correct_tokenizer([text], return_tensors="pt").to(_correct_model.device)
    with torch.no_grad():
        outputs = _correct_model.generate(**inputs, max_new_tokens=1024,
                                          do_sample=False, temperature=0.1)
    gen_text = _correct_tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return gen_text


def correct_sanskrit(noisy_text, model_choice="mistral"):
    """Gradio entry point for AI Sanskrit Post-Correction & Mistake Analysis.

    Supports:
    - 'mistral'   : Mistral 7B / Mistral-Nemo (High accuracy verse cleaner)
    - 'ai4bharat' : AI4Bharat Airavata (IIT Madras Indic model)
    - 'deepseek'  : DeepSeek-R1 (Reasoning Sanskrit reconstruction)
    - 'sarvam'    : Sarvam AI (Indic ligatures & morphology)
    """
    if not (noisy_text or "").strip():
        return json.dumps({"error": "No text provided", "corrected_text": "", "corrections": []})

    import re
    model_id = _resolve_ai_model_id(model_choice)
    model_label = "Mistral-7B" if "mistral" in model_id.lower() else "AI4Bharat" if "ai4bharat" in model_id.lower() else "DeepSeek" if "deepseek" in model_id.lower() else "Sarvam"

    # 1. Mistral API directly if MISTRAL_API_KEY is configured
    mistral_key = os.environ.get("MISTRAL_API_KEY")
    if mistral_key and "mistral" in str(model_choice).lower():
        try:
            import urllib.request
            url = "https://api.mistral.ai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {mistral_key}", "Content-Type": "application/json"}
            payload = json.dumps({
                "model": "mistral-small-latest",
                "messages": [
                    {"role": "system", "content": "You are an expert Sanskrit grammarian and philologist. Analyze noisy Sanskrit OCR text and output pure valid JSON with keys: corrected_text, corrections (list of {original, corrected, reason}), notes."},
                    {"role": "user", "content": f"Correct this noisy Sanskrit OCR text and itemize mistakes:\n{noisy_text}"}
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }).encode('utf-8')
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                res = json.loads(resp.read().decode('utf-8'))
                raw_out = res["choices"][0]["message"]["content"].strip()
                res_obj = json.loads(raw_out)
                res_obj["model_used"] = "Mistral AI (Cloud Agent)"
                return json.dumps(res_obj)
        except Exception as e:
            _log(f"Mistral API failed ({e}) — trying Hugging Face router")

    # 2. Try Hugging Face Serverless Inference API if HF_TOKEN is configured
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            import urllib.request
            url = f"https://router.huggingface.co/hf-inference/models/{model_id}/v1/chat/completions"
            headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
            payload = json.dumps({
                "model": model_id,
                "messages": [
                    {"role": "system", "content": "You are a precise Sanskrit linguistic scholar. Analyze noisy OCR text and output pure valid JSON with keys: corrected_text, corrections (list of {original, corrected, reason}), notes."},
                    {"role": "user", "content": f"Correct this noisy Sanskrit OCR text and list all mistakes:\n{noisy_text}"}
                ],
                "temperature": 0.1,
                "max_tokens": 1024
            }).encode('utf-8')
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                res = json.loads(resp.read().decode('utf-8'))
                raw_out = res["choices"][0]["message"]["content"].strip()
                raw_out = re.sub(r'^```(?:json)?\s*', '', raw_out)
                raw_out = re.sub(r'\s*```$', '', raw_out)
                res_obj = json.loads(raw_out)
                res_obj["model_used"] = model_label
                return json.dumps(res_obj)
        except Exception as e:
            _log(f"HF Inference router failed ({e}) — trying local runner")

    # 3. Local / ZeroGPU runner
    try:
        raw_out = _run_ai_correction_gpu(noisy_text, model_id)
        raw_out = re.sub(r'^```(?:json)?\s*', '', raw_out)
        raw_out = re.sub(r'\s*```$', '', raw_out)
        res_obj = json.loads(raw_out)
        res_obj["model_used"] = model_label
        return json.dumps(res_obj)
    except Exception as e:
        _log(f"AI correction execution failed: {e}")
        return json.dumps({
            "model_used": model_label,
            "corrected_text": noisy_text,
            "corrections": [],
            "notes": f"AI model offline ({str(e)[:80]}). Original text retained.",
            "error": str(e)
        })


# ═══════════════════════════════════════════════════════════════════════════════
# Transcribe entry point (Gradio calls this)
# ═══════════════════════════════════════════════════════════════════════════════

def _page_gray(pil_img):
    if not CV2_AVAILABLE:
        return None
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)


def transcribe(image, engine_choice="auto"):
    """Gradio entry: PIL image + engine radio → result JSON string."""
    t_start = time.time()
    note = None
    try:
        if image is None:
            return json.dumps({"error": "no image received", "engine": None})
        pil = image.convert("RGB") if isinstance(image, PILImage.Image) \
            else PILImage.open(image).convert("RGB")
        medium = analyze_medium(pil)
        want = (engine_choice or "auto").lower()
        # Engine routing:
        # - auto: EasyOCR for whole pages & bounding boxes, MIDF for single lines
        # - easyocr: EasyOCR Devanagari pipeline (hi, mr, ne)
        # - midf: Kraken PP-OCRv6
        # - qwen: Vision LLMs (Qwen 2.5-VL, Gemma 3)
        # - trocr: TrOCR Devanagari
        use_easyocr = (want == "easyocr")
        # "midf-print": MIDF recogniser + OpenCV line finder. Kraken's BLLA
        # segmenter was trained on manuscripts and shreds wide-spaced printed
        # verse (measured 46% vs 26% CER on a Kāvyaprakāśa page), while it is
        # far better on handwriting (23% vs 77% on Mataṅga Bhāratam). So the
        # user picks the layout model per document.
        force_line_mode = (want == "midf-print")
        if force_line_mode:
            want = "midf"
        use_midf = (want in ("midf", "auto") and midf_available()) or \
                   (want == "auto" and not easyocr_available())
        use_qwen = (want == "qwen")

        if use_easyocr:
            try:
                res = easyocr_transcribe_image(pil)
                res["elapsed_s"] = round(time.time() - t_start, 1)
                return json.dumps(res)
            except Exception as e:
                _log(f"easyocr failed: {traceback.format_exc(limit=3)}")
                note = (f"EasyOCR failed ({str(e)[:120]}) — answered with "
                        f"{'MIDF' if midf_available() else 'TrOCR'} instead.")
                use_midf = midf_available()

        gray = _page_gray(pil)
        lines = segment_lines(gray) if gray is not None else [
            {"bbox": {"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0},
             "crop": None}]
        engine_used = None

        if use_qwen or want == "qwen":
            if not vision_available():
                if want == "qwen":
                    # No GPU here: MIDF is a far better stand-in than TrOCR
                    # (measured CER 0.05 vs 0.88 on Devanagari lines).
                    use_midf = midf_available()
                    note = ("Vision model needs a GPU (this Space has none "
                            "right now) — answered with "
                            f"{'MIDF' if use_midf else 'TrOCR'} instead.")
                use_qwen = False
            else:
                try:
                    full = _vision_transcribe_gpu(pil)
                    qlines = [l.strip() for l in full.splitlines()]
                    while qlines and not qlines[0]:
                        qlines.pop(0)
                    while qlines and not qlines[-1]:
                        qlines.pop()
                    if len(qlines) > len(lines) and lines:
                        head, tail = qlines[:len(lines) - 1], qlines[len(lines) - 1:]
                        qlines = head + [" ".join(t for t in tail if t)]
                    out_lines = []
                    for i, ln in enumerate(lines):
                        raw = qlines[i] if i < len(qlines) else ""
                        out_lines.append({"text": raw, "confidence": None,
                                          "bbox": ln["bbox"],
                                          "granularity": "page"})
                    engine_used = _vision_engine_name(_vision_mtype)
                    return json.dumps({
                        "text": "\n".join(l["text"] for l in out_lines),
                        "lines": out_lines, "medium": medium, "engine": engine_used,
                        "note": note,
                        "elapsed_s": round(time.time() - t_start, 1)})
                except Exception as e:
                    _log(f"vision model failed: {traceback.format_exc(limit=3)}")
                    # Fall back to MIDF (CER ~0.05 on lines), never TrOCR (~0.88).
                    use_midf = midf_available()
                    note = (f"Vision model failed ({str(e)[:120]}) — "
                            f"answered with {'MIDF' if use_midf else 'TrOCR'} instead.")

        # ── MIDF line path (default): one line at a time, CPU, ~0.2 s ─────────
        if use_midf:
            try:
                out_lines = []
                page_lines = None
                if (MIDF_MODE == "page" and not force_line_mode
                        and load_midf() and _midf_seg is not None):
                    try:
                        page_lines = midf_page_lines(pil)
                    except Exception as e:
                        _log(f"MIDF page mode failed, using line mode: {e}")
                if page_lines:
                    for pl in page_lines:
                        out_lines.append({
                            "text": pl["text"], "confidence": pl["confidence"],
                            "bbox": pl["bbox"],
                            "words": [{"text": pl["text"], "approx": True,
                                       "bbox": pl["bbox"]}],
                            "granularity": "line"})
                elif load_midf():
                    crops = [(PILImage.fromarray(ln["crop"]).convert("RGB")
                              if ln["crop"] is not None and CV2_AVAILABLE else pil)
                             for ln in lines]
                    for ln, (txt, conf) in zip(
                            lines, midf_recognize_lines([_midf_prepare(c)
                                                         for c in crops])):
                        words = []
                        crop = ln["crop"]
                        if crop is not None and CV2_AVAILABLE:
                            spans = [(w["x0"], w["x1"])
                                  for w in split_words(crop, MIDF_WORD_GAP)]
                            cw = max(crop.shape[1], 1)
                            for (x0, x1), t in zip(spans, apportion_words(txt, spans)):
                                f0, f1 = x0 / cw, x1 / cw
                                words.append({
                                    "text": t, "approx": True,
                                    "bbox": {
                                        "xmin": ln["bbox"]["xmin"] + f0 * (ln["bbox"]["xmax"] - ln["bbox"]["xmin"]),
                                        "ymin": ln["bbox"]["ymin"],
                                        "xmax": ln["bbox"]["xmin"] + f1 * (ln["bbox"]["xmax"] - ln["bbox"]["xmin"]),
                                        "ymax": ln["bbox"]["ymax"]}})
                        if not words:
                            words = [{"text": txt, "approx": True, "bbox": ln["bbox"]}]
                        elif MIDF_SPACES:
                            txt = " ".join(w["text"] for w in words).strip()
                        out_lines.append({"text": txt, "confidence": conf,
                                          "bbox": ln["bbox"], "words": words,
                                          "granularity": "line"})
                if out_lines:
                    extra = ("Word boxes are approximate — MIDF recognises the "
                             "line as one string, without spaces.")
                    return json.dumps({
                        "text": "\n".join(l["text"] for l in out_lines),
                        "lines": out_lines, "medium": medium,
                        "engine": "midf-ppocrv6",
                        "note": f"{note} {extra}".strip() if note else extra,
                        "elapsed_s": round(time.time() - t_start, 1)})
            except Exception as e:
                _log(f"midf failed: {traceback.format_exc(limit=3)}")
                note = (f"MIDF failed ({str(e)[:120]}) — answered with TrOCR "
                        "instead.")

        # ── TrOCR word path ──
        load_trocr()
        jobs = []  # (line_idx, word_crop, strip_w, line_bbox)
        for li, ln in enumerate(lines):
            crop = ln["crop"]
            if crop is None or not CV2_AVAILABLE:
                jobs.append((li, None, 1, ln["bbox"]))
                continue
            for w in split_words(crop):
                jobs.append((li, w["crop"], crop.shape[1], ln["bbox"]))
        line_texts = [[] for _ in lines]
        line_words = [[] for _ in lines]
        line_confs = [[] for _ in lines]
        for i in range(0, len(jobs), TROCR_BATCH):
            chunk = jobs[i:i + TROCR_BATCH]
            pils = [_word_to_pil(c) if c is not None
                    else pil.resize((224, 224)) for _, c, _, _ in chunk]
            try:
                texts, confs = trocr_decode(pils)
            except Exception as e:
                _log(f"trocr batch failed ({e}) — retrying singly")
                texts, confs = [], []
                for p in pils:
                    try:
                        t, c = trocr_decode([p])
                        texts += t
                        confs += c
                    except Exception:
                        texts.append("")
                        confs.append(None)
            for (li, _, sw, lb), t, c in zip(chunk, texts, confs):
                # word bbox: re-derive x-fractions from the job order is
                # unreliable here, so store line-fraction boxes per word index
                line_texts[li].append(t)
                if c is not None:
                    line_confs[li].append(c)
        # rebuild word boxes in a second pass (cheap, keeps code simple)
        out_lines = []
        for li, ln in enumerate(lines):
            crop = ln["crop"]
            words = []
            if crop is not None and CV2_AVAILABLE:
                for w, t in zip(split_words(crop), line_texts[li]):
                    f0 = w["x0"] / max(crop.shape[1], 1)
                    f1 = w["x1"] / max(crop.shape[1], 1)
                    words.append({"text": t, "bbox": {
                        "xmin": ln["bbox"]["xmin"] + f0 * (ln["bbox"]["xmax"] - ln["bbox"]["xmin"]),
                        "ymin": ln["bbox"]["ymin"],
                        "xmax": ln["bbox"]["xmin"] + f1 * (ln["bbox"]["xmax"] - ln["bbox"]["xmin"]),
                        "ymax": ln["bbox"]["ymax"]}})
            else:
                words = [{"text": t, "bbox": ln["bbox"]} for t in line_texts[li]]
            raw_line = " ".join(t for t in line_texts[li] if t).strip()
            vis = (sum(line_confs[li]) / len(line_confs[li])) if line_confs[li] else None
            out_lines.append({"text": raw_line, "confidence": vis,
                              "bbox": ln["bbox"], "words": words,
                              "granularity": "word"})
        return json.dumps({
            "text": "\n".join(l["text"] for l in out_lines),
            "lines": out_lines, "medium": medium,
            "engine": "trocr-devanagari", "note": note,
            "elapsed_s": round(time.time() - t_start, 1)})
    except Exception as e:
        _log(f"transcribe failed: {traceback.format_exc(limit=5)}")
        return json.dumps({"error": str(e)[:500], "engine": None})


# ═══════════════════════════════════════════════════════════════════════════════
# Gradio UI + API
# ═══════════════════════════════════════════════════════════════════════════════
import gradio as gr  # noqa: E402

# ── Gradio ↔ ZeroGPU auto-wrap hardening ─────────────────────────────────────
# Gradio 6.27 wraps every event handler with spaces.gradio_auto_wrap() whenever
# it believes it is on a Space. Depending on who imported `spaces` first, that
# symbol can be missing and the app dies while building the UI:
#   AttributeError: module 'spaces' has no attribute 'gradio_auto_wrap'
# We decorate the GPU call ourselves (see gpu_decorator), so auto-wrap buys us
# nothing — make the call site inert and re-check `spaces` once gradio is loaded.
try:
    import spaces as _sp_late
    if not HAS_SPACES or spaces.__class__.__name__ == "_DummySpaces":
        spaces = _sp_late          # the real module, if it is importable now
        HAS_SPACES = True
    if not hasattr(spaces, "gradio_auto_wrap"):
        spaces.gradio_auto_wrap = lambda task, *a, **k: task
    if hasattr(spaces, "disable_gradio_auto_wrap"):
        spaces.disable_gradio_auto_wrap()
except Exception as _e:
    _log(f"spaces re-import failed ({_e}) — running without ZeroGPU")
try:
    from gradio.block_function import BlockFunction
    BlockFunction.spaces_auto_wrap = lambda self: None   # never call into spaces
except Exception as _e:
    _log(f"could not neutralise gradio auto-wrap ({_e})")
try:
    _bf = sys.modules.get("gradio.block_function")
    _log(f"spaces={'real' if HAS_SPACES else 'stub'} "
         f"gradio_auto_wrap={hasattr(spaces, 'gradio_auto_wrap')} "
         f"gradio_spaces={'None' if getattr(_bf, 'spaces', None) is None else 'module'}")
except Exception:
    pass

with gr.Blocks(title="EyM Sanskrit Cloud HTR") as demo:
    gr.Markdown(
        "# 🕉️ EyM Sanskrit Cloud HTR\n"
        "Zero-install neural backend for the EyM Scholar's Workstation. "
        "**MIDF Sanskrit OCR** (Kraken PP-OCRv6, 17 MB, CPU — default), "
        "**Qwen2.5-VL Sanskrit OCR** (GPU, whole pages and hard cases) and "
        "**TrOCR-Devanagari** (CPU fallback). Upload a manuscript page — or "
        "call the `/predict` API straight from the EyM site.")
    with gr.Row():
        img_in = gr.Image(type="pil", label="Manuscript page / line / word")
        with gr.Column():
            engine_in = gr.Radio(["auto", "easyocr", "midf", "midf-print", "qwen", "trocr"],
                                 value="auto", label="Engine",
                                 info="auto = EasyOCR (whole page) / MIDF (lines); "
                                      "easyocr = EasyOCR Devanagari; midf = Kraken PP-OCRv6; "
                                      "qwen = Vision LLMs (GPU); trocr = TrOCR")
            go = gr.Button("🪔 Transcribe", variant="primary")
    out = gr.Textbox(label="result_json", lines=3,
                     info="JSON: {text, lines[], engine, medium, note}")
    go.click(fn=transcribe, inputs=[img_in, engine_in], outputs=out,
             api_name="predict")

    with gr.Accordion("✨ AI Sanskrit Post-Correction & Mistake Analysis (Mistral / AI4Bharat)", open=False):
        with gr.Row():
            ai_model_in = gr.Dropdown(
                ["mistral", "ai4bharat", "deepseek", "sarvam"],
                value="mistral",
                label="AI Model",
                info="Mistral-7B (Sanskrit Verse Cleaner); AI4Bharat (IIT Madras Indic); DeepSeek (Reasoning)"
            )
        ai_in = gr.Textbox(label="Raw Sanskrit OCR Text", lines=3,
                           placeholder="Paste or transcribe Sanskrit text to analyze...")
        ai_btn = gr.Button("✨ AI Correct & Analyze Mistakes", variant="secondary")
        ai_out = gr.Textbox(label="correction_json", lines=4,
                            info="JSON: {model_used, corrected_text, corrections[], notes}")
        ai_btn.click(fn=correct_sanskrit, inputs=[ai_in, ai_model_in], outputs=ai_out,
                     api_name="correct_sanskrit")

    gr.Markdown(
        "_MIDF is 17 MB and runs on CPU (~0.2 s per line). First-ever Qwen use "
        "downloads ~15GB (minutes); sleeping Spaces wake on first request "
        "(~30–60s)._")

if __name__ == "__main__":
    _log("starting EyM Cloud HTR Space …")
    # MIDF is tiny (17 MB) — warm it in the background so the first request
    # is instant. TrOCR stays the always-on fallback and loads as before.
    try:
        if midf_available():
            threading.Thread(target=load_midf, daemon=True).start()
        else:
            _log("MIDF off (kraken not installed or EYM_MIDF=0)")
    except Exception as e:
        _log(f"MIDF preload thread failed: {e}")
    try:
        if os.environ.get("EYM_TROCR_PRELOAD", "1") != "0":
            load_trocr()
    except Exception as e:
        _log(f"TrOCR preload failed (will retry per request): {e}")
    demo.queue(max_size=8).launch()
