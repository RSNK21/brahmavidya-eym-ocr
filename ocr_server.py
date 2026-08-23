"""
EyM Sanskrit OCR — Parinamika Model Server
=========================================
Flask backend for the Sanskrit-OCR-main (Parinamika) seq2seq OCR model.

Endpoints:
  GET  /health        → {"status":"ok","engine":"parinamika"|"tesseract"}
  POST /ocr           → {"text":..., "lines":[...], "engine":...}
  POST /segment       → {"lines":[{bbox, image_b64},...]}

Model: Download from https://drive.google.com/file/d/1KJ6vORY-Ybi_ldvdj2cDGAYsnRG4wdCR/view
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

# ── OpenCV / PIL (always required) ────────────────────────────────────────────
try:
    import cv2
    import numpy as np
    from PIL import Image as PILImage
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    log.warning("cv2/PIL not available — line segmentation disabled")

# ── Tesseract fallback ────────────────────────────────────────────────────────
try:
    import pytesseract
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
        raise RuntimeError("Neither Parinamika nor system Tesseract is available")
    pil_img = PILImage.open(io.BytesIO(image_bytes))
    data = pytesseract.image_to_data(pil_img, lang=lang, output_type=pytesseract.Output.DICT)
    text = " ".join(w for w in data["text"] if w.strip())
    return {"text": text, "confidence": None}


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


# ═══════════════════════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "engine": ENGINE,
        "model_ready": _model_ready,
        "model_error": _model_error,
        "cv2": CV2_AVAILABLE,
        "tesseract": TESSERACT_AVAILABLE,
    })


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
    Run OCR on an image (line or page).
    Accepts: multipart/form-data 'image', or JSON 'image_b64'.
    Optional: 'lang' (default: 'san'), 'mode' ('line'|'page', default 'page')
    Returns: {"text":..., "lines":[...], "engine":..., "confidence":...}
    """
    try:
        image_bytes = _get_image_bytes(request)
        lang = (request.form.get("lang") or
                (request.json or {}).get("lang", "san"))
        mode = (request.form.get("mode") or
                (request.json or {}).get("mode", "page"))

        if mode == "page":
            # Segment into lines first, then run OCR on each
            lines = segment_lines_opencv(image_bytes)
            if not lines:
                lines = [{"image_b64": base64.b64encode(image_bytes).decode(), "bbox": None}]
            full_text_parts = []
            line_results = []
            for line_info in lines:
                lb = base64.b64decode(line_info["image_b64"])
                try:
                    if _model_ready:
                        pred = predict_parinamika(lb)
                    else:
                        pred = predict_tesseract(lb, lang)
                except Exception as line_err:
                    pred = {"text": "", "confidence": None, "error": str(line_err)}
                full_text_parts.append(pred["text"])
                line_results.append({**pred, "bbox": line_info["bbox"]})
            return jsonify({
                "text": "\n".join(full_text_parts),
                "lines": line_results,
                "engine": ENGINE,
            })
        else:
            # Single line mode
            if _model_ready:
                pred = predict_parinamika(image_bytes)
            else:
                pred = predict_tesseract(image_bytes, lang)
            return jsonify({**pred, "engine": ENGINE, "lines": []})

    except Exception as e:
        log.error("OCR error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


def _get_image_bytes(req) -> bytes:
    """Extract raw image bytes from either multipart or JSON request."""
    if "image" in req.files:
        return req.files["image"].read()
    if req.is_json:
        data = req.json or {}
        b64 = data.get("image_b64") or data.get("image")
        if b64:
            # Strip data URI prefix if present
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            return base64.b64decode(b64)
    raise ValueError("No image provided. Send 'image' as file upload or 'image_b64' in JSON.")


# ═══════════════════════════════════════════════════════════════════════════════
# Startup
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("EyM OCR Server starting up …")
    try_load_parinamika()
    if not _model_ready:
        log.info("Running in Tesseract fallback mode (Parinamika model not loaded)")
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=False)
