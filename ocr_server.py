"""
EyM Sanskrit OCR — Model Server
===============================
Flask backend for EyM's OCR/HTR pipeline. Tries engines in this priority
order, falling back automatically if a given engine isn't available:

  1. Neural HTR ("qwen-vl" or "trocr-devanagari") — real Devanagari/Sanskrit
     handwriting models, far more accurate than the two below on cursive
     manuscript hands. See NEURAL ENGINE section below to configure.
  2. Parinamika — the legacy TF1.15 seq2seq model this server originally
     shipped with (kept for backward compatibility; most installs won't
     have this).
  3. Tesseract — last-resort fallback via system pytesseract.

Endpoints:
  GET  /health        → {"status":"ok","engine":"qwen-vl"|"trocr-devanagari"|"parinamika"|"tesseract", ...}
  POST /ocr           → {"text":..., "lines":[...], "engine":...}
  POST /segment       → {"lines":[{bbox, image_b64},...]}

NEURAL ENGINE (new):
  Set the EYM_NEURAL_ENGINE environment variable to choose a backend:
    EYM_NEURAL_ENGINE=qwen    → Sanskrit-fine-tuned Qwen2.5-VL-7B (best accuracy,
                                 needs a GPU with ~16GB VRAM, or EYM_QWEN_4BIT=1
                                 for a quantized ~6-8GB footprint)
    EYM_NEURAL_ENGINE=trocr   → TrOCR fine-tuned on handwritten Devanagari words
                                 (much lighter; runs on CPU, just slower)
    EYM_NEURAL_ENGINE=auto    → try qwen, fall back to trocr, fall back to
                                 Parinamika/Tesseract (default)
    EYM_NEURAL_ENGINE=none    → skip neural models entirely (old behaviour)
  Model checkpoints are downloaded from Hugging Face on first run and cached
  locally (~/.cache/huggingface) — no manual download step needed, unlike
  the Parinamika model below.

  EYM_QWEN_MODEL_ID   (default: diabolic6045/Sanskrit-Qwen2.5-VL-7B-Instruct-OCR)
  EYM_TROCR_MODEL_ID  (default: paudelanil/trocr-devanagari-2)

Legacy Parinamika model: Download from
  https://drive.google.com/file/d/1KJ6vORY-Ybi_ldvdj2cDGAYsnRG4wdCR/view
Place extracted files in: modelss/ (next to this script)

Requirements: pip install -r requirements_ocr.txt
"""

import os
import sys
import json
import base64
import logging
import io
import traceback
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("eym-ocr-server")

app = Flask(__name__)
CORS(app, origins=["*"])  # Allow browser tool (any origin during local dev)

# ── Model state ────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "modelss"
_model = None        # Parinamika TF model (loaded lazily)
_sess = None         # TF session
_model_ready = False
_model_error = None
ENGINE = "tesseract"  # default; becomes "parinamika" when model loads

# ── Attempt to import TF / Parinamika ─────────────────────────────────────────
try:
    # Parinamika model lives in Sanskrit-OCR-main/mysite/main/model
    PARINAMIKA_PATH = Path(__file__).parent.parent / \
        "Sanskrit-OCR-main" / "Sanskrit-OCR-main" / "mysite" / "main"
    if PARINAMIKA_PATH.exists():
        sys.path.insert(0, str(PARINAMIKA_PATH))
        from model.model import Model as PaM
        import tensorflow as tf
        TF_AVAILABLE = True
        log.info("TensorFlow import OK")
    else:
        TF_AVAILABLE = False
        log.warning("Parinamika path not found: %s", PARINAMIKA_PATH)
except Exception as e:
    TF_AVAILABLE = False
    log.warning("TensorFlow not available: %s", e)

# ── Neural HTR engine (Qwen2.5-VL Sanskrit fine-tune / TrOCR-Devanagari) ──────
# These are optional — the server still runs Parinamika/Tesseract-only if
# torch/transformers aren't installed. See requirements_ocr.txt.
try:
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from transformers import VisionEncoderDecoderModel, TrOCRProcessor
    TORCH_AVAILABLE = True
    log.info("torch/transformers import OK (cuda available: %s)", torch.cuda.is_available())
except Exception as e:
    TORCH_AVAILABLE = False
    log.warning("torch/transformers not available — neural HTR engine disabled: %s", e)

NEURAL_ENGINE_PREF = os.environ.get("EYM_NEURAL_ENGINE", "auto").lower()  # auto | qwen | trocr | none
QWEN_MODEL_ID = os.environ.get("EYM_QWEN_MODEL_ID", "diabolic6045/Sanskrit-Qwen2.5-VL-7B-Instruct-OCR")
TROCR_MODEL_ID = os.environ.get("EYM_TROCR_MODEL_ID", "paudelanil/trocr-devanagari-2")
QWEN_4BIT = os.environ.get("EYM_QWEN_4BIT", "0") == "1"

_neural_ready = False
_neural_error = None
_qwen_model = None
_qwen_processor = None
_trocr_model = None
_trocr_processor = None

# ── OpenCV / PIL (always required) ────────────────────────────────────────────
try:
    import cv2
    import numpy as np
    from PIL import Image as PILImage
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    log.warning("cv2/PIL not available — line segmentation disabled")

# ── PyMuPDF (PDF rasterization) ────────────────────────────────────────────────
try:
    import pymupdf as fitz
    FITZ_AVAILABLE = True
    log.info("PyMuPDF available for server-side PDF processing")
except ImportError:
    try:
        import fitz
        FITZ_AVAILABLE = True
        log.info("PyMuPDF (fitz) available for server-side PDF processing")
    except ImportError:
        FITZ_AVAILABLE = False
        log.warning("PyMuPDF not available — server-side PDF rasterization disabled")


# ── Tesseract fallback ────────────────────────────────────────────────────────
try:
    import pytesseract
    if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
    log.info("Tesseract system install found")
except Exception:
    TESSERACT_AVAILABLE = False
    log.warning("System Tesseract not found — OCR fallback unavailable")


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def try_load_parinamika():
    """Load the Parinamika model if TF + model files are available."""
    global _model, _sess, _model_ready, _model_error, ENGINE
    if not TF_AVAILABLE:
        _model_error = "TensorFlow 1.15 not installed"
        return
    if not MODEL_DIR.exists():
        _model_error = f"Model directory not found: {MODEL_DIR}"
        log.warning(_model_error)
        return
    checkpoint_files = list(MODEL_DIR.glob("*.index"))
    if not checkpoint_files:
        _model_error = f"No checkpoint files found in {MODEL_DIR}"
        log.warning(_model_error)
        return
    try:
        log.info("Loading Parinamika model from %s …", MODEL_DIR)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        _sess = tf.Session(config=config)
        _model = PaM(
            phase="predict",
            visualize=False,
            output_dir="results",
            batch_size=1,
            initial_learning_rate=1.0,
            steps_per_checkpoint=0,
            model_dir=str(MODEL_DIR) + "\\",
            target_embedding_size=10,
            attn_num_hidden=128,
            attn_num_layers=2,
            clip_gradients=True,
            max_gradient_norm=5.0,
            session=_sess,
            load_model=True,
            gpu_id=0,
            use_gru=False,
            use_distance=True,
            max_image_width=3200,
            max_image_height=150,
            max_prediction_length=600,
            channels=1,
        )
        _model_ready = True
        ENGINE = "parinamika"
        log.info("Parinamika model loaded successfully ✓")
    except Exception as e:
        _model_error = str(e)
        log.error("Failed to load Parinamika model: %s", e)


def try_load_qwen_vl():
    """Load the Sanskrit-fine-tuned Qwen2.5-VL model (best accuracy, needs GPU)."""
    global _qwen_model, _qwen_processor, _neural_ready, _neural_error, ENGINE
    if not TORCH_AVAILABLE:
        _neural_error = "torch/transformers not installed"
        return False
    try:
        log.info("Loading Qwen2.5-VL Sanskrit OCR model (%s) — this downloads "
                  "the checkpoint from Hugging Face on first run …", QWEN_MODEL_ID)
        load_kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
            if QWEN_4BIT:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            log.warning("No CUDA GPU detected — Qwen2.5-VL will run on CPU and be slow. "
                        "Consider EYM_NEURAL_ENGINE=trocr for CPU use instead.")
        _qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
        _qwen_model = AutoModelForImageTextToText.from_pretrained(QWEN_MODEL_ID, **load_kwargs)
        if not torch.cuda.is_available():
            _qwen_model = _qwen_model.to("cpu")
        _qwen_model.eval()
        _neural_ready = True
        ENGINE = "qwen-vl"
        log.info("Qwen2.5-VL Sanskrit OCR model loaded successfully ✓")
        return True
    except Exception as e:
        _neural_error = str(e)
        log.error("Failed to load Qwen2.5-VL model: %s", e)
        return False


def try_load_trocr():
    """Load the TrOCR-Devanagari handwriting model (lighter, CPU-friendly)."""
    global _trocr_model, _trocr_processor, _neural_ready, _neural_error, ENGINE
    if not TORCH_AVAILABLE:
        _neural_error = "torch/transformers not installed"
        return False
    try:
        log.info("Loading TrOCR-Devanagari model (%s) — this downloads the "
                  "checkpoint from Hugging Face on first run …", TROCR_MODEL_ID)
        _trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_ID)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _trocr_model = _trocr_model.to(device)
        _trocr_model.eval()
        _neural_ready = True
        ENGINE = "trocr-devanagari"
        log.info("TrOCR-Devanagari model loaded successfully ✓ (device: %s)", device)
        return True
    except Exception as e:
        _neural_error = str(e)
        log.error("Failed to load TrOCR-Devanagari model: %s", e)
        return False


def try_load_neural_engine():
    """Load a neural HTR engine per EYM_NEURAL_ENGINE. Tries qwen -> trocr in
    'auto' mode; a specific choice ('qwen' or 'trocr') is not cross-tried."""
    if NEURAL_ENGINE_PREF == "none":
        log.info("EYM_NEURAL_ENGINE=none — skipping neural HTR engine")
        return False
    if NEURAL_ENGINE_PREF == "qwen":
        return try_load_qwen_vl()
    if NEURAL_ENGINE_PREF == "trocr":
        return try_load_trocr()
    # auto: prefer the stronger model, fall back to the lighter one
    if try_load_qwen_vl():
        return True
    log.info("Qwen2.5-VL unavailable, trying TrOCR-Devanagari instead …")
    return try_load_trocr()


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def decode_parinamika_output(out: str) -> str:
    """
    Decode the Parinamika model's raw output string into Unicode text.
    The model outputs sequences of Unicode codepoints as decimal strings.
    Ported from Sanskrit-OCR-main/mysite/main/predict.py lines 62-75.
    """
    out_word = ""
    i = 0
    while i < len(out):
        # 4-digit codepoints: 23xx, 24xx (Devanagari range 0x0900-0x097F)
        if out[i:i+2] in ('23', '24'):
            try:
                out_word += chr(int(out[i:i+4]))
                i += 4
                continue
            except (ValueError, OverflowError):
                break
        # 2-digit ASCII codepoints: digits, space, punctuation
        chunk2 = out[i:i+2]
        if (chunk2 in ('32', '35', '95', '46', '44', '45') or
                ('48' <= chunk2 <= '57')):
            try:
                out_word += chr(int(chunk2))
                i += 2
                continue
            except (ValueError, OverflowError):
                break
        # 3-digit: 124 = |
        if out[i:i+3] == '124':
            out_word += chr(124)
            i += 3
            continue
        # Unknown — stop
        break
    return out_word


def predict_parinamika(image_bytes: bytes) -> dict:
    """Run Parinamika model prediction on a raw image bytes object."""
    if not _model_ready or _model is None:
        raise RuntimeError("Parinamika model not loaded")
    out, probability = _model.predict(image_bytes)
    text = decode_parinamika_output(out)
    return {"text": text, "confidence": float(probability) if probability else None}


def predict_tesseract(image_bytes: bytes, lang: str = "san") -> dict:
    """Fallback: run system Tesseract on image bytes."""
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("Neither a neural engine nor system Tesseract is available")
    pil_img = PILImage.open(io.BytesIO(image_bytes))
    data = pytesseract.image_to_data(pil_img, lang=lang, output_type=pytesseract.Output.DICT)
    text = " ".join(w for w in data["text"] if w.strip())
    return {"text": text, "confidence": None}


QWEN_PROMPT = (
    "Transcribe the Sanskrit/Devanagari text visible in this image exactly as "
    "written, including any handwriting. Output only the transcribed Devanagari "
    "text with no commentary, translation, or extra formatting."
)


def predict_qwen_vl(image_bytes: bytes) -> dict:
    """Run the Sanskrit-fine-tuned Qwen2.5-VL model on a line/page image."""
    if not _neural_ready or _qwen_model is None or ENGINE != "qwen-vl":
        raise RuntimeError("Qwen2.5-VL model not loaded")
    pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": QWEN_PROMPT},
        ],
    }]
    prompt_text = _qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _qwen_processor(text=[prompt_text], images=[pil_img], return_tensors="pt")
    inputs = {k: v.to(_qwen_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        generated = _qwen_model.generate(**inputs, max_new_tokens=256, do_sample=False)
    # Only decode the newly generated tokens, not the echoed prompt.
    input_len = inputs["input_ids"].shape[1]
    output_ids = generated[:, input_len:]
    text = _qwen_processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()
    return {"text": text, "confidence": None}


def predict_trocr(image_bytes: bytes) -> dict:
    """Run the TrOCR-Devanagari model on a line/word-crop image.

    TrOCR is a word/short-line recognizer, not a document-level model — it
    performs best on a single line or word crop, which is exactly the
    granularity the /ocr endpoint's page-mode segmentation already produces.
    """
    if not _neural_ready or _trocr_model is None or ENGINE != "trocr-devanagari":
        raise RuntimeError("TrOCR-Devanagari model not loaded")
    pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    pixel_values = _trocr_processor(images=pil_img, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(_trocr_model.device)
    with torch.no_grad():
        generated_ids = _trocr_model.generate(pixel_values, max_new_tokens=64)
    text = _trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return {"text": text, "confidence": None}


def predict_neural(image_bytes: bytes) -> dict:
    """Dispatch to whichever neural engine is currently loaded."""
    if ENGINE == "qwen-vl":
        return predict_qwen_vl(image_bytes)
    if ENGINE == "trocr-devanagari":
        return predict_trocr(image_bytes)
    raise RuntimeError("No neural engine is loaded")


# ═══════════════════════════════════════════════════════════════════════════════
# Line segmentation (ported from Sanskrit-OCR-main/mysite/main/boxes.py)
# ═══════════════════════════════════════════════════════════════════════════════

def segment_lines_opencv(image_bytes: bytes) -> list:
    """
    Segment a page image into individual text lines using horizontal
    projection profile. Ported from boxes.py::segment_images().

    Returns list of dicts: {bbox: {xmin,ymin,xmax,ymax}, image_b64: str}
    """
    if not CV2_AVAILABLE:
        return []
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    h, w = img.shape

    # Binarize
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection profile
    proj = np.sum(thresh, axis=1)

    # Find line boundaries where projection > threshold
    LINE_THRESH = w * 5  # pixels per row to be considered "text"
    in_line = False
    lines = []
    y_start = 0
    for y in range(h):
        if proj[y] > LINE_THRESH and not in_line:
            y_start = y
            in_line = True
        elif proj[y] <= LINE_THRESH and in_line:
            if y - y_start > 8:  # minimum line height
                lines.append((y_start, y))
            in_line = False
    if in_line and h - y_start > 8:
        lines.append((y_start, h))

    # If no lines found, treat whole image as one line
    if not lines:
        lines = [(0, h)]

    results = []
    PADDING = 4
    for (y1, y2) in lines:
        y1p = max(0, y1 - PADDING)
        y2p = min(h, y2 + PADDING)
        crop = img[y1p:y2p, 0:w]
        # Encode cropped line as base64 PNG
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            continue
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        results.append({
            "bbox": {
                "xmin": 0.0,
                "ymin": float(y1p) / h,
                "xmax": 1.0,
                "ymax": float(y2p) / h
            },
            "image_b64": b64,
        })
    return results


import sqlite3
import unicodedata
import hashlib

# ── SQLite Database Setup ──────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "eym_manuscript_htr.db"

def init_db():
    """Initialize SQLite tables for manuscript profiles, confusions, and audit history."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS manuscripts (
                id TEXT PRIMARY KEY,
                title TEXT,
                script TEXT DEFAULT 'Devanagari',
                medium TEXT DEFAULT 'paper_ink',
                period TEXT DEFAULT 'unknown',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scribal_confusions (
                from_char TEXT,
                to_char TEXT,
                weight REAL DEFAULT 1.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (from_char, to_char)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS corrections_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                manuscript_id TEXT,
                word_original TEXT,
                word_corrected TEXT,
                medium TEXT,
                reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS active_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT,
                original_text TEXT,
                accepted_text TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Seed default Devanagari scribal confusions if empty
        cursor.execute("SELECT COUNT(*) FROM scribal_confusions")
        if cursor.fetchone()[0] == 0:
            default_confusions = [
                ("व", "ब", 0.85), ("श", "ष", 0.78), ("ि", "ी", 0.65),
                ("ं", "ँ", 0.60), ("न", "ण", 0.70), ("ध", "घ", 0.72),
                ("ख", "रव", 0.80), ("त्त", "त्र", 0.68)
            ]
            cursor.executemany("INSERT INTO scribal_confusions (from_char, to_char, weight) VALUES (?, ?, ?)", default_confusions)
        conn.commit()
        conn.close()
        log.info("SQLite HTR Database initialized at %s ✓", DB_PATH)
    except Exception as e:
        log.error("Failed to initialize SQLite database: %s", e)

# Run DB init
init_db()

# ── Load Sanskrit Lexicon for Language Model Ranking ─────────────────────────
HEADWORDS_PATH = Path(__file__).parent / "sanskrit-headwords.json"
SANSKRIT_LEXICON = set()
if HEADWORDS_PATH.exists():
    try:
        with open(HEADWORDS_PATH, "r", encoding="utf-8") as f:
            hw_data = json.load(f)
            if isinstance(hw_data, list):
                SANSKRIT_LEXICON = set(hw_data)
            elif isinstance(hw_data, dict):
                SANSKRIT_LEXICON = set(hw_data.keys())
        log.info("Loaded Sanskrit Lexicon with %d headwords ✓", len(SANSKRIT_LEXICON))
    except Exception as lex_err:
        log.warning("Could not parse Sanskrit headwords lexicon: %s", lex_err)


# ── Document Medium & Degradation Analysis ───────────────────────────────────

def analyze_medium_and_degradation(image_bytes: bytes) -> dict:
    """
    Analyze manuscript image to classify writing medium and degradation level.
    Supported classes: paper_ink, palm_leaf_incised, palm_leaf_ink, stone_inscription, degraded_paper
    """
    if not CV2_AVAILABLE:
        return {
            "medium_class": "paper_ink",
            "confidence": 0.80,
            "degradation_score": 0.15,
            "details": {"reason": "CV2 unavailable — using default paper profile"}
        }

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"medium_class": "paper_ink", "confidence": 0.75, "degradation_score": 0.20}

    h, w, _ = img.shape
    aspect_ratio = float(w) / float(h)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    # Calculate color variance (palm leaf / aged paper has high warm yellow/brown channel bias)
    b, g, r = cv2.split(img)
    rg_diff = float(np.mean(np.abs(r.astype(float) - g.astype(float))))
    rb_diff = float(np.mean(np.abs(r.astype(float) - b.astype(float))))

    # Edge analysis for incised lines vs ink strokes
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    medium_class = "paper_ink"
    confidence = 0.88
    degradation_score = min(1.0, max(0.0, (128.0 - std_val) / 128.0))

    # Classification Heuristics:
    if aspect_ratio > 3.2 and mean_val < 160 and rb_diff > 25:
        medium_class = "palm_leaf_incised"
        confidence = 0.92
    elif aspect_ratio > 2.8 and (rb_diff > 20 or rg_diff > 10):
        medium_class = "palm_leaf_ink"
        confidence = 0.90
    elif std_val < 32 and mean_val < 140:
        medium_class = "stone_inscription"
        confidence = 0.86
    elif mean_val < 150 or std_val > 65:
        medium_class = "degraded_paper"
        confidence = 0.87
        degradation_score = min(1.0, degradation_score + 0.35)

    return {
        "medium_class": medium_class,
        "confidence": round(confidence, 2),
        "degradation_score": round(degradation_score, 2),
        "details": {
            "aspect_ratio": round(aspect_ratio, 2),
            "mean_luminance": round(mean_val, 1),
            "contrast_std": round(std_val, 1),
            "color_bias_rb": round(rb_diff, 1),
            "edge_density": round(edge_density, 3)
        }
    }


# ── N-Best HTR Decoding & Sanskrit Re-ranking ────────────────────────────────

def generate_nbest_candidates(text: str, visual_conf: float = 0.85) -> list:
    """
    Generate N-best candidate readings using Sanskrit lexicon & scribal confusion matrix.
    Returns list of dicts: [{text, visual_conf, language_conf, final_score, is_lexical}]
    """
    if not text or not text.strip():
        return []

    # NFC Normalisation
    norm_text = unicodedata.normalize("NFC", text.strip())
    candidates = []
    
    # Candidate 1: Direct Visual Recognition
    is_in_lexicon = norm_text in SANSKRIT_LEXICON
    lang_conf = 0.95 if is_in_lexicon else 0.40
    vis_conf = visual_conf if visual_conf is not None else 0.80
    final_score = round(0.6 * vis_conf + 0.4 * lang_conf, 3)

    candidates.append({
        "text": norm_text,
        "diplomatic_text": text,
        "visual_confidence": round(vis_conf, 2),
        "language_confidence": round(lang_conf, 2),
        "final_score": final_score,
        "is_lexical": is_in_lexicon,
        "source": "visual_htr"
    })

    # Generate lookalike candidates from confusion matrix
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT from_char, to_char, weight FROM scribal_confusions")
        confusions = cursor.fetchall()
        conn.close()

        for from_c, to_c, weight in confusions:
            if from_c in norm_text:
                variant = norm_text.replace(from_c, to_c)
                if variant != norm_text:
                    var_in_lex = variant in SANSKRIT_LEXICON
                    var_lang_conf = 0.96 if var_in_lex else 0.35
                    var_vis_conf = max(0.20, vis_conf - (1.0 - weight) * 0.3)
                    var_final = round(0.55 * var_vis_conf + 0.45 * var_lang_conf, 3)
                    candidates.append({
                        "text": variant,
                        "diplomatic_text": variant,
                        "visual_confidence": round(var_vis_conf, 2),
                        "language_confidence": round(var_lang_conf, 2),
                        "final_score": var_final,
                        "is_lexical": var_in_lex,
                        "source": f"confusion_{from_c}->{to_c}"
                    })
    except Exception as c_err:
        log.warning("Error looking up scribal confusions: %s", c_err)

    # Sort candidates by final_score descending
    candidates.sort(key=lambda c: c["final_score"], reverse=True)
    return candidates[:5]


# ═══════════════════════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "engine": ENGINE,
        # "model_ready" here reflects whichever engine is actually active,
        # so the front-end's existing check (data.engine + data.model_ready)
        # keeps working unmodified whether that's neural, Parinamika, or none.
        "model_ready": _neural_ready or _model_ready,
        "model_error": _neural_error if not _neural_ready else _model_error,
        "neural_ready": _neural_ready,
        "neural_error": _neural_error,
        "neural_engine_pref": NEURAL_ENGINE_PREF,
        "torch_available": TORCH_AVAILABLE,
        "cv2": CV2_AVAILABLE,
        "tesseract": TESSERACT_AVAILABLE,
        "fitz": FITZ_AVAILABLE,
        "db_ready": DB_PATH.exists(),
        "lexicon_words": len(SANSKRIT_LEXICON)
    })


def _rasterize_pdf_page(pdf_bytes: bytes, page_num: int = 0) -> bytes:
    if not FITZ_AVAILABLE:
        raise RuntimeError("PDF uploaded but PyMuPDF (fitz) is not installed on the server.")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_num < 0 or page_num >= len(doc):
        page_num = 0
    page = doc.load_page(page_num)
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def _get_image_bytes(req) -> bytes:
    """Extract raw image bytes from either multipart or JSON request, converting PDF to PNG if needed."""
    raw = None
    if "image" in req.files:
        raw = req.files["image"].read()
    elif req.is_json:
        data = req.json or {}
        b64 = data.get("image_b64") or data.get("image")
        if b64:
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            raw = base64.b64decode(b64)

    if not raw:
        raise ValueError("No image provided. Send 'image' as file upload or 'image_b64' in JSON.")

    if raw.startswith(b"%PDF-"):
        page_str = req.form.get("page") or (req.json or {}).get("page", 0)
        try:
            page_num = int(page_str)
        except (ValueError, TypeError):
            page_num = 0
        return _rasterize_pdf_page(raw, page_num)

    return raw


@app.route("/analyze_document", methods=["POST"])
def analyze_document():
    """Analyze manuscript image for writing medium, degradation, and line count."""
    try:
        image_bytes = _get_image_bytes(request)
        medium_info = analyze_medium_and_degradation(image_bytes)
        lines = segment_lines_opencv(image_bytes)
        return jsonify({
            **medium_info,
            "line_count": len(lines),
            "status": "success"
        })
    except Exception as e:
        log.error("Document analysis error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/segment", methods=["POST"])
def segment():
    """
    Segment a full page image into lines.
    Accepts: multipart/form-data with field 'image', or JSON with 'image_b64'.
    Returns: {"lines": [{bbox, image_b64}, ...]}
    """
    try:
        image_bytes = _get_image_bytes(request)
        lines = segment_lines_opencv(image_bytes)
        return jsonify({"lines": lines, "count": len(lines)})
    except Exception as e:
        log.error("Segment error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/ocr", methods=["POST"])
def ocr():
    """
    Run HTR/OCR on an image (line or page).
    Returns: {"text":..., "lines":[...], "medium":..., "engine":..., "confidence":...}
    """
    try:
        image_bytes = _get_image_bytes(request)
        lang = (request.form.get("lang") or
                (request.json or {}).get("lang", "san"))
        mode = (request.form.get("mode") or
                (request.json or {}).get("mode", "page"))

        # Analyze writing medium
        medium_analysis = analyze_medium_and_degradation(image_bytes)

        if mode == "page":
            lines = segment_lines_opencv(image_bytes)
            if not lines:
                lines = [{"image_b64": base64.b64encode(image_bytes).decode(), "bbox": None}]
            full_text_parts = []
            line_results = []
            for line_info in lines:
                lb = base64.b64decode(line_info["image_b64"])
                try:
                    if _neural_ready:
                        pred = predict_neural(lb)
                    elif _model_ready:
                        pred = predict_parinamika(lb)
                    else:
                        pred = predict_tesseract(lb, lang)
                except Exception as line_err:
                    pred = {"text": "", "confidence": None, "error": str(line_err)}
                
                raw_text = pred.get("text", "")
                nbest = generate_nbest_candidates(raw_text, pred.get("confidence"))
                top_reading = nbest[0]["text"] if nbest else unicodedata.normalize("NFC", raw_text)
                
                disagreement = (
                    len(nbest) > 1 and 
                    nbest[0]["text"] != raw_text and 
                    nbest[0]["is_lexical"]
                )

                full_text_parts.append(top_reading)
                line_results.append({
                    **pred,
                    "text": top_reading,
                    "diplomatic_text": raw_text,
                    "normalised_text": top_reading,
                    "nbest_candidates": nbest,
                    "visual_language_disagreement": disagreement,
                    "bbox": line_info["bbox"]
                })
            
            return jsonify({
                "text": "\n".join(full_text_parts),
                "lines": line_results,
                "medium": medium_analysis,
                "engine": ENGINE,
            })
        else:
            if _neural_ready:
                pred = predict_neural(image_bytes)
            elif _model_ready:
                pred = predict_parinamika(image_bytes)
            else:
                pred = predict_tesseract(image_bytes, lang)
            
            raw_text = pred.get("text", "")
            nbest = generate_nbest_candidates(raw_text, pred.get("confidence"))
            top_reading = nbest[0]["text"] if nbest else unicodedata.normalize("NFC", raw_text)

            return jsonify({
                **pred,
                "text": top_reading,
                "diplomatic_text": raw_text,
                "normalised_text": top_reading,
                "nbest_candidates": nbest,
                "medium": medium_analysis,
                "engine": ENGINE,
                "lines": []
            })

    except Exception as e:
        log.error("OCR error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/manuscript/feedback", methods=["POST"])
def manuscript_feedback():
    """
    Log scholar correction feedback to SQLite and update scribal confusions.
    Accepts JSON: {original_text, accepted_text, medium, reason}
    """
    try:
        data = request.json or {}
        orig = data.get("original_text", "").strip()
        accepted = data.get("accepted_text", "").strip()
        medium = data.get("medium", "paper_ink")
        reason = data.get("reason", "scholarly_correction")

        if not orig or not accepted:
            return jsonify({"error": "original_text and accepted_text required"}), 400

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Log audit entry
        cursor.execute(
            "INSERT INTO corrections_audit (manuscript_id, word_original, word_corrected, medium, reason) VALUES (?, ?, ?, ?, ?)",
            ("EYM-MSS-LOCAL", orig, accepted, medium, reason)
        )

        # Update scribal confusion matrix if single-char mismatch
        if len(orig) == len(accepted) == 1 and orig != accepted:
            cursor.execute("""
                INSERT INTO scribal_confusions (from_char, to_char, weight)
                VALUES (?, ?, 1.0)
                ON CONFLICT(from_char, to_char) DO UPDATE SET
                weight = weight + 0.5,
                updated_at = CURRENT_TIMESTAMP
            """, (orig, accepted))

        conn.commit()
        conn.close()

        return jsonify({
            "status": "success",
            "message": "Feedback saved to manuscript SQLite store ✓",
            "original": orig,
            "accepted": accepted
        })
    except Exception as e:
        log.error("Feedback endpoint error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/manuscript/confusions", methods=["GET"])
def get_confusions():
    """Return active scribal confusion matrix from SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT from_char, to_char, weight, updated_at FROM scribal_confusions ORDER BY weight DESC")
        rows = cursor.fetchall()
        conn.close()

        confusions = [
            {"from": r[0], "to": r[1], "weight": round(r[2], 2), "updated_at": r[3]}
            for r in rows
        ]
        return jsonify({"confusions": confusions, "count": len(confusions)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# Startup
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("EyM OCR & Devanagari HTR Server starting up …")
    if not try_load_neural_engine():
        log.info("No neural engine loaded (%s) — trying legacy Parinamika model …", _neural_error)
        try_load_parinamika()
    if not (_neural_ready or _model_ready):
        log.info("Running in Tesseract fallback mode (no neural or Parinamika model loaded)")
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=False)

