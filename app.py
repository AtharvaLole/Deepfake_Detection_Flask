import os
import uuid
import hashlib
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import cv2
import pytesseract
from PIL import Image
import numpy as np
import soundfile as sf
import pytesseract
from flask import Flask, request, jsonify, render_template_string

import platform
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"




app = Flask(__name__)
CORS(app)

#configu
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "temp_uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

ALLOWED_IMAGE_EXT = {"jpg", "jpeg", "png", "bmp", "tiff"}
ALLOWED_DOC_EXT   = {"pdf"}
ALLOWED_VIDEO_EXT = {"mp4", "mov", "avi", "mkv"}
ALLOWED_AUDIO_EXT = {"wav", "mp3", "flac", "m4a"}


#upload files path joined
def save_upload(file_storage):
    """Save an uploaded file to UPLOAD_FOLDER with a random name."""
    filename = secure_filename(file_storage.filename)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    new_name = f"{uuid.uuid4().hex}.{ext}" if ext else uuid.uuid4().hex
    path = os.path.join(app.config["UPLOAD_FOLDER"], new_name)
    file_storage.save(path)
    return path, ext

#if any error with upload
def error(msg, status=400):
    return jsonify({"success": False, "error": msg}), status

# -------------------------
# Generic helpers for "models"
# -------------------------

def file_hash_score(path: str) -> float:
    """
    Deterministically map file bytes -> score in [0, 1).
    We use this sometimes to jitter thresholds so different files
    don't always sit exactly on the same boundary.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    digest = h.digest()
    val = int.from_bytes(digest[:4], "big")  # first 4 bytes
    return (val % 1000) / 1000.0


def compute_blur_score(image: np.ndarray) -> float:
    """
    Simple blur metric: variance of Laplacian.
    Higher value => sharper image, lower => more blurry.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(fm)


#responses of your model
#basically jo apka model detect karega usko yaha pe likhenge and the necessary response will be provided

def model_image_text(path, file_type):
    """
    Baseline model for 'manipulated text in image/docs'.

    Strategy:
    - Run OCR with Tesseract
    - Look at OCR confidence distribution
    - Very inconsistent or generally low-confidence text is flagged as 'suspicious/manipulated'.

    This is NOT real forensic detection, but:
    - It uses actual OCR on content
    - Different images/docs => different scores
    """

    # Load image
    if file_type == "document":
    # PDF support requires Poppler, which is not installed on this machine.
    # Return a clean error instead of crashing with 500.
        return {
            "success": False,
            "detector_type": "image_text",
            "error": "PDF document analysis is not available on this server (Poppler not installed). Please upload image files (JPG/PNG) for now."
        }
    else:
        img = cv2.imread(path)


    if img is None:
        return {
            "success": False,
            "detector_type": "image_text",
            "error": "Could not load image/document"
        }

    # Run Tesseract OCR with data output
    ocr_data = pytesseract.image_to_data(
        img,
        output_type=pytesseract.Output.DICT
    )

    confs = [int(c) for c in ocr_data.get("conf", []) if c != "-1"]

    if not confs:
        avg_conf = 0.0
        low_conf_ratio = 1.0
    else:
        confs_arr = np.array(confs)
        avg_conf = float(np.mean(confs_arr))
        low_conf_ratio = float(np.mean(confs_arr < 50))

    # Score: combine low confidence ratio + hash-based jitter
    base_score = (low_conf_ratio + (100 - avg_conf) / 100.0) / 2.0
    jitter = file_hash_score(path) * 0.2  # +/- small change
    score = float(max(0.0, min(1.0, base_score * 0.8 + jitter)))

    if score < 0.3:
        verdict = "real"
    elif score < 0.6:
        verdict = "suspicious"
    else:
        verdict = "manipulated"

    text = " ".join(ocr_data.get("text", []))[:1000]  # partial text

    return {
        "success": True,
        "detector_type": "image_text",
        "verdict": verdict,
        "score": score,
        "details": {
            "file_type": file_type,
            "avg_ocr_confidence": avg_conf,
            "low_confidence_ratio": low_conf_ratio,
            "sample_extracted_text": text,
            "note": "Heuristic OCR-based check, not a true tampering detector."
        }
    }



def model_video_deepfake(path):
    """
    Baseline 'video deepfake' heuristic.

    Strategy:
    - Sample frames every N steps
    - Compute color histogram differences between consecutive frames
    - Very low variation or very high spikiness => 'suspicious/manipulated'

    NOT a true deepfake model, but actually processes the video.
    """

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {
            "success": False,
            "detector_type": "video_deepfake",
            "error": "Could not open video"
        }

    hist_diffs = []
    frame_count = 0
    prev_hist = None

    step = 10  # analyze every 10th frame
    ok, frame = cap.read()
    while ok:
        frame_count += 1
        if frame_count % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                hist_diffs.append(diff)

            prev_hist = hist

        ok, frame = cap.read()

    cap.release()

    if not hist_diffs:
        return {
            "success": True,
            "detector_type": "video_deepfake",
            "verdict": "inconclusive",
            "score": 0.5,
            "details": {
                "frames_analyzed": 0,
                "note": "Video too short or unreadable."
            }
        }

    diffs = np.array(hist_diffs)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs))

    # Heuristic scoring:
    # - too smooth (very low diff) or too jittery (very high diff) => suspicious
    base_score = abs(mean_diff - 0.3) + std_diff  # arbitrary centre
    base_score = min(base_score, 1.0)
    jitter = file_hash_score(path) * 0.1
    score = float(max(0.0, min(1.0, base_score * 0.8 + jitter)))

    if score < 0.3:
        verdict = "real"
    elif score < 0.6:
        verdict = "suspicious"
    else:
        verdict = "manipulated"

    return {
        "success": True,
        "detector_type": "video_deepfake",
        "verdict": verdict,
        "score": score,
        "details": {
            "frames_analyzed": int(frame_count / step),
            "mean_hist_difference": mean_diff,
            "std_hist_difference": std_diff,
            "note": "Histogram-based heuristic; not a real deepfake classifier."
        }
    }



# You need a Haar cascade file. Download:
# https://github.com/opencv/opencv/tree/master/data/haarcascades
# e.g. haarcascade_frontalface_default.xml
# Place it in your project dir and set the path:
HAAR_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH) if os.path.exists(HAAR_CASCADE_PATH) else None


def model_face_image(path):
    """
    Baseline 'face manipulation / synthetic' heuristic.

    Strategy:
    - Detect faces with Haar cascade
    - Measure blur
    - Use number of faces + blur to mark suspicious/manipulated.

    NOT a real GAN/synthetic detection, but uses real image features.
    """
    img = cv2.imread(path)
    if img is None:
        return {
            "success": False,
            "detector_type": "face_image",
            "error": "Could not load image"
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = []
    if face_cascade is not None:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    blur_score = compute_blur_score(img)
    img_h, img_w = img.shape[:2]

    # Simple heuristics:
    # - no faces => suspicious
    # - too many faces => suspicious
    # - extremely sharp or extremely blurry => suspicious/manipulated

    num_faces = len(faces)
    base = 0.0

    if num_faces == 0:
        base += 0.4
    elif num_faces > 3:
        base += 0.3
    else:
        base += 0.1

    # Normalize blur: typical thresholds
    if blur_score < 50:
        base += 0.3  # very blurry
    elif blur_score > 500:
        base += 0.2  # suspiciously sharp

    base = min(base, 1.0)
    jitter = file_hash_score(path) * 0.2
    score = float(max(0.0, min(1.0, base * 0.8 + jitter)))

    if score < 0.3:
        verdict = "real"
    elif score < 0.6:
        verdict = "suspicious"
    else:
        verdict = "manipulated"

    faces_info = [
        {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        for (x, y, w, h) in faces
    ]

    return {
        "success": True,
        "detector_type": "face_image",
        "verdict": verdict,
        "score": score,
        "details": {
            "image_width": img_w,
            "image_height": img_h,
            "num_faces_detected": num_faces,
            "faces": faces_info,
            "blur_score": blur_score,
            "note": "Haar face + blur heuristic; not a true synthetic-face detector."
        }
    }

def model_voice_clone(path):
    """
    Baseline 'cloned voice' heuristic without librosa.

    Strategy:
    - Load audio with soundfile (supports wav, flac, ogg; mp3 if backend allows)
    - Compute:
        - zero-crossing rate (ZCR)
        - spectral flatness (geometric mean / arithmetic mean of magnitude spectrum)
    - Simple rules on these features.

    Still NOT a real clone detector, but much lighter than librosa/numba.
    """
    try:
        # y: np.ndarray, sr: sample rate
        y, sr = sf.read(path, always_2d=False)
    except Exception as e:
        print("[VOICE ERROR]", e)
        return {
            "success": False,
            "detector_type": "voice_clone",
            "error": f"Could not read audio: {e}"
        }

    # Convert stereo -> mono if needed
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1)

    if not isinstance(y, np.ndarray) or y.size == 0:
        return {
            "success": True,
            "detector_type": "voice_clone",
            "verdict": "inconclusive",
            "score": 0.5,
            "details": {"note": "Empty or unreadable audio."}
        }

    y = y.astype(float)
    duration_sec = float(len(y) / sr) if sr else 0.0

    # --- Zero Crossing Rate (simple manual version) ---
    # Sign-based zero crossing count / total frames
    signs = np.sign(y)
    signs[signs == 0] = 1  # avoid zero plateaus
    zero_crossings = np.where(np.diff(signs))[0]
    zcr_mean = float(len(zero_crossings) / max(1, len(y)))

    # --- Spectral Flatness (simple single-frame estimate) ---
    # Use a window from the start (or shorter if file is tiny)
    N = min(2048, len(y))
    if N < 64:
        # Too short for reliable spectrum, fall back to neutral
        flatness_mean = 0.5
    else:
        window = y[:N]
        spectrum = np.abs(np.fft.rfft(window)) + 1e-12  # avoid log(0)
        geo_mean = float(np.exp(np.mean(np.log(spectrum))))
        arith_mean = float(np.mean(spectrum))
        flatness_mean = float(geo_mean / arith_mean) if arith_mean > 0 else 0.5

    # Heuristic:
    # - extremely low ZCR + very low flatness => monotone / synthetic-like
    # - extremely high ZCR or flatness => noisy / artificial
    base = 0.0

    if zcr_mean < 0.02:
        base += 0.3
    elif zcr_mean > 0.15:
        base += 0.2

    if flatness_mean < 0.2:
        base += 0.2
    elif flatness_mean > 0.7:
        base += 0.2

    base = min(base, 1.0)
    jitter = file_hash_score(path) * 0.2
    score = float(max(0.0, min(1.0, base * 0.8 + jitter)))

    if score < 0.3:
        verdict = "real"
    elif score < 0.6:
        verdict = "suspicious"
    else:
        verdict = "manipulated"

    return {
        "success": True,
        "detector_type": "voice_clone",
        "verdict": verdict,
        "score": score,
        "details": {
            "duration_seconds": duration_sec,
            "zero_crossing_rate_mean": zcr_mean,
            "spectral_flatness_mean": flatness_mean,
            "note": "Audio-feature heuristic without librosa; not a production-grade cloned-voice detector."
        }
    }

#status of api
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


#img/doc manipulation
@app.route("/api/detect/image-text", methods=["POST"])
def detect_image_text():
    if "file" not in request.files:
        return error("Missing file field 'file'")

    file = request.files["file"]
    if file.filename == "":
        return error("No file selected")

    path, ext = save_upload(file)
    ext = ext.lower()

    # classify type
    if ext in ALLOWED_IMAGE_EXT:
        file_type = "image"
    elif ext in ALLOWED_DOC_EXT:
        file_type = "document"
    else:
        os.remove(path)
        return error(f"Unsupported extension for image/doc: .{ext}. Please upload .jpeg, .png, .jpg, .bmp, .tiff files only")

    try:
        response = model_image_text(path, file_type)
    finally:
        if os.path.exists(path):
            os.remove(path)

    return jsonify(response)

#video detection call
@app.route("/api/detect/video", methods=["POST"])
def detect_video():
    if "file" not in request.files:
        return error("Missing file field 'file'")

    file = request.files["file"]
    if file.filename == "":
        return error("No file selected")

    path, ext = save_upload(file)
    ext = ext.lower()

    if ext not in ALLOWED_VIDEO_EXT:
        os.remove(path)
        return error(f"Unsupported video extension: .{ext}. Please upload .mp4, .mkv, .mov, .avi files only.")

    try:
        response = model_video_deepfake(path)
    finally:
        if os.path.exists(path):
            os.remove(path)

    return jsonify(response)


#face-detection call 
@app.route("/api/detect/face-image", methods=["POST"])
def detect_face_image():
    if "file" not in request.files:
        return error("Missing file field 'file'")

    file = request.files["file"]
    if file.filename == "":
        return error("No file selected")

    path, ext = save_upload(file)
    ext = ext.lower()

    if ext not in ALLOWED_IMAGE_EXT:
        os.remove(path)
        return error(f"Unsupported image extension: .{ext}. Please upload .jpeg, .png, .jpg, .bmp, .tiff")

    try:
        response = model_face_image(path)
    finally:
        if os.path.exists(path):
            os.remove(path)

    return jsonify(response)


#audio voice clone
@app.route("/api/detect/voice", methods=["POST"])
def detect_voice():
    if "file" not in request.files:
        return error("Missing file field 'file'")

    file = request.files["file"]
    if file.filename == "":
        return error("No file selected")

    path, ext = save_upload(file)
    ext = ext.lower()

    if ext not in ALLOWED_AUDIO_EXT:
        os.remove(path)
        return error(f"Unsupported audio extension: .{ext}. Please upload only .wav, .mp3, .flac, .m4a files only")

    try:
        response = model_voice_clone(path)
    finally:
        if os.path.exists(path):
            os.remove(path)

    return jsonify(response)
DEMO_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>Deepfake Detection Dashboard</title>
  <style>
    :root {
      --bg-main: #0b1120;
      --bg-card: #020617;
      --border-card: #1e293b;
      --accent: #3b82f6;
      --accent-soft: rgba(59, 130, 246, 0.15);
      --text-main: #e5e7eb;
      --text-muted: #9ca3af;
      --text-subtle: #6b7280;
      --green: #22c55e;
      --amber: #eab308;
      --red: #ef4444;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(59,130,246,0.22), transparent 55%),
        radial-gradient(circle at bottom right, rgba(236,72,153,0.18), transparent 55%),
        var(--bg-main);
      color: var(--text-main);
      padding: 24px 16px;
      display: flex;
      justify-content: center;
    }

    .shell {
      width: 100%;
      max-width: 1100px;
      background: rgba(15,23,42,0.96);
      border-radius: 20px;
      border: 1px solid rgba(148,163,184,0.35);
      box-shadow: 0 18px 45px rgba(15,23,42,0.9);
      padding: 20px 22px 24px;
      backdrop-filter: blur(18px);
    }

    h1 {
      margin: 0 0 4px 0;
      font-size: 22px;
      letter-spacing: 0.02em;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .subtitle {
      color: var(--text-muted);
      font-size: 13px;
      margin-bottom: 16px;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 18px;
    }

    .card {
      border: 1px solid var(--border-card);
      padding: 14px 14px 16px;
      border-radius: 14px;
      box-shadow: 0 10px 25px rgba(15,23,42,0.8);
      background: radial-gradient(circle at top left, rgba(15,23,42,0.96), rgba(15,23,42,0.98));
      position: relative;
      overflow: hidden;
    }

    .card::before {
      content: "";
      position: absolute;
      inset: -1px;
      border-radius: inherit;
      background: linear-gradient(135deg, rgba(59,130,246,0.22), transparent 60%);
      opacity: 0;
      transition: opacity 0.16s ease-out;
      pointer-events: none;
    }

    .card:hover::before {
      opacity: 1;
    }

    .card h2 {
      margin: 0 0 4px 0;
      font-size: 16px;
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 10px;
    }

    .card p {
      margin: 0 0 10px 0;
      font-size: 12px;
      color: var(--text-subtle);
    }

    form {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-bottom: 6px;
    }

    input[type="file"] {
      font-size: 11px;
      color: var(--text-main);
      background: rgba(15,23,42,0.9);
      border-radius: 10px;
      padding: 7px 9px;
      border: 1px dashed rgba(148,163,184,0.8);
      cursor: pointer;
    }

    input[type="file"]::file-selector-button {
      margin-right: 8px;
      border: none;
      border-radius: 999px;
      padding: 4px 10px;
      background: rgba(37,99,235,0.95);
      color: #e5e7eb;
      font-size: 11px;
      cursor: pointer;
    }

    input[type="file"]:hover::file-selector-button {
      background: rgba(59,130,246,1);
    }

    button {
      align-self: flex-start;
      padding: 6px 14px;
      border-radius: 999px;
      border: none;
      background: linear-gradient(135deg, #2563eb, #4f46e5);
      color: white;
      font-size: 12px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      box-shadow: 0 10px 24px rgba(37,99,235,0.4);
      transition: transform 120ms ease-out, box-shadow 120ms ease-out, filter 120ms ease-out;
    }

    button::after {
      content: "⟶";
      font-size: 12px;
    }

    button:hover {
      transform: translateY(-1px);
      box-shadow: 0 14px 30px rgba(37,99,235,0.55);
      filter: brightness(1.05);
    }

    button:active {
      transform: translateY(0) scale(0.99);
      box-shadow: 0 8px 18px rgba(15,23,42,0.9);
    }

    .status {
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 4px;
    }

    .status.error {
      color: #fecaca;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 999px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin: 8px 6px 2px 0;
      border: 1px solid transparent;
    }

    .badge-real {
      background: rgba(22,163,74,0.12);
      color: #bbf7d0;
      border-color: rgba(34,197,94,0.9);
    }

    .badge-suspicious {
      background: rgba(234,179,8,0.14);
      color: #facc15;
      border-color: rgba(234,179,8,0.95);
    }

    .badge-manipulated {
      background: rgba(248,113,113,0.16);
      color: #fecaca;
      border-color: rgba(248,113,113,0.95);
    }

    .badge-inconclusive {
      background: rgba(148,163,184,0.2);
      color: #e5e7eb;
      border-color: rgba(148,163,184,0.95);
    }

    .score-line,
    .risk-line {
      font-size: 11px;
      margin-top: 2px;
      color: var(--text-muted);
    }

    .score-line span,
    .risk-line span {
      font-weight: 500;
      margin-left: 4px;
    }

    .risk-line.risk-low span {
      color: var(--green);
    }

    .risk-line.risk-medium span {
      color: var(--amber);
    }

    .risk-line.risk-high span {
      color: var(--red);
    }

    .risk-meter {
      margin-top: 8px;
    }

    .risk-meter-label {
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--text-subtle);
      margin-bottom: 4px;
    }

    .risk-meter-bar {
      width: 100%;
      height: 8px;
      border-radius: 999px;
      background: rgba(15,23,42,0.9);
      border: 1px solid rgba(31,41,55,0.9);
      overflow: hidden;
    }

    .risk-meter-fill {
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #22c55e, #eab308, #ef4444);
      transition: width 0.2s ease-out;
    }

    small.helper {
      display: block;
      margin-top: 6px;
      font-size: 10px;
      color: var(--text-subtle);
    }

    @media (max-width: 768px) {
      .shell {
        padding: 18px 14px 20px;
        border-radius: 18px;
      }
    }
    /* --- Global blocking loader overlay --- */
    .loading-overlay {
      position: fixed;
      inset: 0;
      display:none;
      background: rgba(2, 6, 23, 0.78);
      backdrop-filter: blur(6px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 9999;
    }

    .loading-overlay.active {
      display: flex;
    }

    .loading-card {
      width: min(420px, calc(100% - 32px));
      border: 1px solid rgba(148, 163, 184, 0.35);
      background: rgba(15, 23, 42, 0.92);
      border-radius: 16px;
      padding: 18px 16px;
      box-shadow: 0 18px 45px rgba(15,23,42,0.9);
      text-align: center;
    }

    .spinner {
      width: 44px;
      height: 44px;
      border-radius: 999px;
      border: 4px solid rgba(148,163,184,0.25);
      border-top-color: rgba(59,130,246,0.95);
      margin: 0 auto 12px;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .loading-title {
      font-size: 14px;
      color: var(--text-main);
      margin: 0 0 4px 0;
      letter-spacing: 0.02em;
    }

    .loading-subtitle {
      font-size: 12px;
      color: var(--text-muted);
      margin: 0;
    }

    body.is-loading {
      overflow: hidden;
    }

    /* Disable pointer events while loading (belt + suspenders) */
    body.is-loading * {
      pointer-events: none;
    }
    body.is-loading .loading-overlay,
    body.is-loading .loading-overlay * {
      pointer-events: all;
    }

  </style>
  <script>
  // Ensure loader is OFF by default on page load (prevents stuck overlay on reload/back nav)
  document.addEventListener("DOMContentLoaded", function () {
    document.body.classList.remove("is-loading");
  });
</script>

</head>
<body>
<div id="loadingOverlay" class="loading-overlay" aria-live="polite" aria-busy="true">
  <div class="loading-card">
    <div class="spinner"></div>
    <p class="loading-title">Analyzing your file…</p>
    <p class="loading-subtitle" id="loadingSubtitle">Please wait. This can take a few seconds.</p>
  </div>
</div>


<div class="shell">
  <h1>Deepfake Detection Dashboard</h1>
  <div class="subtitle">
    Upload media files to assess manipulation risk across images, video, face photos, and voice recordings.<br>
    Supported files:<br>
    Image: .jpeg, .jpg, .png, .bmp, .tiff <br>
    Video: .mov, .mp4, .mkv, .avi <br>
    Face: .jpeg, .jpg, .png, .bmp, .tiff <br>
    Audio: .mp3, .wav, .m4a, .flac <br>


  </div>

  <div class="grid">

    <!-- Image / Document -->
    <div class="card">
      <h2>Image Manipulation</h2>
      <p>Analyze images or documents for potentially manipulated text.</p>
      <form method="post" enctype="multipart/form-data">
        <input type="hidden" name="module" value="image-text">
        <input type="file" name="file" required>
        <button type="submit">Upload & Analyze</button>
      </form>
      {% if image_text %}
        {% if not image_text.success %}
          <div class="status error"><strong>Error:</strong> {{ image_text.error }}</div>
        {% else %}
          <div class="status">Analysis complete.</div>
          <div>
            {% set verdict = image_text.verdict or "inconclusive" %}
            {% set score = image_text.score if image_text.score is not none else None %}
            {% if verdict.lower() == "real" %}
              {% set badge_class = "badge-real" %}
            {% elif verdict.lower() == "manipulated" %}
              {% set badge_class = "badge-manipulated" %}
            {% elif verdict.lower() == "suspicious" %}
              {% set badge_class = "badge-suspicious" %}
            {% else %}
              {% set badge_class = "badge-inconclusive" %}
            {% endif %}
            <span class="badge {{ badge_class }}">{{ verdict.upper() }}</span>

            <div class="score-line">
              Score:
              <span>
              {% if score is not none %}
                {{ (score * 100) | round(1) }}%
              {% else %}
                N/A
              {% endif %}
              </span>
            </div>

            <div class="risk-meter">
              <div class="risk-meter-label">Risk level</div>
              <div class="risk-meter-bar">
                <div class="risk-meter-fill"
                  style="width: {% if score is not none %}{{ (score * 100) | round(0) }}{% else %}0{% endif %}%;">
                </div>
              </div>
            </div>

            <div class="risk-line
              {% if score is not none and score < 0.33 %}risk-low{% elif score is not none and score < 0.66 %}risk-medium{% elif score is not none %}risk-high{% endif %}
            ">
              Risk:
              <span>
              {% if score is none %}
                N/A
              {% elif score < 0.33 %}
                LOW
              {% elif score < 0.66 %}
                MEDIUM
              {% else %}
                HIGH
              {% endif %}
              </span>
            </div>
          </div>
        {% endif %}
      {% endif %}
      <small class="helper">Text-based anomaly scoring for uploaded images and documents.</small>
    </div>

    <!-- Video -->
    <div class="card">
      <h2>Video Deepfake</h2>
      <p>Scan videos for abnormal frame patterns that may indicate manipulation.</p>
      <form method="post" enctype="multipart/form-data">
        <input type="hidden" name="module" value="video">
        <input type="file" name="file" required>
        <button type="submit">Upload & Analyze</button>
      </form>
      {% if video %}
        {% if not video.success %}
          <div class="status error"><strong>Error:</strong> {{ video.error }}</div>
        {% else %}
          <div class="status">Analysis complete.</div>
          {% set verdict = video.verdict or "inconclusive" %}
          {% set score = video.score if video.score is not none else None %}
          {% if verdict.lower() == "real" %}
            {% set badge_class = "badge-real" %}
          {% elif verdict.lower() == "manipulated" %}
            {% set badge_class = "badge-manipulated" %}
          {% elif verdict.lower() == "suspicious" %}
            {% set badge_class = "badge-suspicious" %}
          {% else %}
            {% set badge_class = "badge-inconclusive" %}
          {% endif %}
          <span class="badge {{ badge_class }}">{{ verdict.upper() }}</span>

          <div class="score-line">
            Score:
            <span>
            {% if score is not none %}
              {{ (score * 100) | round(1) }}%
            {% else %}
              N/A
            {% endif %}
            </span>
          </div>

          <div class="risk-meter">
            <div class="risk-meter-label">Risk level</div>
            <div class="risk-meter-bar">
              <div class="risk-meter-fill"
                style="width: {% if score is not none %}{{ (score * 100) | round(0) }}{% else %}0{% endif %}%;">
              </div>
            </div>
          </div>

          <div class="risk-line
            {% if score is not none and score < 0.33 %}risk-low{% elif score is not none and score < 0.66 %}risk-medium{% elif score is not none %}risk-high{% endif %}
          ">
            Risk:
            <span>
            {% if score is none %}
              N/A
            {% elif score < 0.33 %}
              LOW
            {% elif score < 0.66 %}
              MEDIUM
            {% else %}
              HIGH
            {% endif %}
            </span>
          </div>
        {% endif %}
      {% endif %}
      <small class="helper">Frame-level pattern analysis for uploaded videos.</small>
    </div>

    <!-- Face Image -->
    <div class="card">
      <h2>Face Image</h2>
      <p>Evaluate face images using detection and blur-based integrity checks.</p>
      <form method="post" enctype="multipart/form-data">
        <input type="hidden" name="module" value="face-image">
        <input type="file" name="file" required>
        <button type="submit">Upload & Analyze</button>
      </form>
      {% if face_image %}
        {% if not face_image.success %}
          <div class="status error"><strong>Error:</strong> {{ face_image.error }}</div>
        {% else %}
          <div class="status">Analysis complete.</div>
          {% set verdict = face_image.verdict or "inconclusive" %}
          {% set score = face_image.score if face_image.score is not none else None %}
          {% if verdict.lower() == "real" %}
            {% set badge_class = "badge-real" %}
          {% elif verdict.lower() == "manipulated" %}
            {% set badge_class = "badge-manipulated" %}
          {% elif verdict.lower() == "suspicious" %}
            {% set badge_class = "badge-suspicious" %}
          {% else %}
            {% set badge_class = "badge-inconclusive" %}
          {% endif %}
          <span class="badge {{ badge_class }}">{{ verdict.upper() }}</span>

          <div class="score-line">
            Score:
            <span>
            {% if score is not none %}
              {{ (score * 100) | round(1) }}%
            {% else %}
              N/A
            {% endif %}
            </span>
          </div>

          <div class="risk-meter">
            <div class="risk-meter-label">Risk level</div>
            <div class="risk-meter-bar">
              <div class="risk-meter-fill"
                style="width: {% if score is not none %}{{ (score * 100) | round(0) }}{% else %}0{% endif %}%;">
              </div>
            </div>
          </div>

          <div class="risk-line
            {% if score is not none and score < 0.33 %}risk-low{% elif score is not none and score < 0.66 %}risk-medium{% elif score is not none %}risk-high{% endif %}
          ">
            Risk:
            <span>
            {% if score is none %}
              N/A
            {% elif score < 0.33 %}
              LOW
            {% elif score < 0.66 %}
              MEDIUM
            {% else %}
              HIGH
            {% endif %}
            </span>
          </div>
        {% endif %}
      {% endif %}
      <small class="helper">Face detection and blur scoring for uploaded images.</small>
    </div>

    <!-- Voice -->
    <div class="card">
      <h2>Voice Clone</h2>
      <p>Check audio recordings for patterns commonly seen in synthetic or cloned voices.</p>
      <form method="post" enctype="multipart/form-data">
        <input type="hidden" name="module" value="voice">
        <input type="file" name="file" required>
        <button type="submit">Upload & Analyze</button>
      </form>
      {% if voice %}
        {% if not voice.success %}
          <div class="status error"><strong>Error:</strong> {{ voice.error }}</div>
        {% else %}
          <div class="status">Analysis complete.</div>
          {% set verdict = voice.verdict or "inconclusive" %}
          {% set score = voice.score if voice.score is not none else None %}
          {% if verdict.lower() == "real" %}
            {% set badge_class = "badge-real" %}
          {% elif verdict.lower() == "manipulated" %}
            {% set badge_class = "badge-manipulated" %}
          {% elif verdict.lower() == "suspicious" %}
            {% set badge_class = "badge-suspicious" %}
          {% else %}
            {% set badge_class = "badge-inconclusive" %}
          {% endif %}
          <span class="badge {{ badge_class }}">{{ verdict.upper() }}</span>

          <div class="score-line">
            Score:
            <span>
            {% if score is not none %}
              {{ (score * 100) | round(1) }}%
            {% else %}
              N/A
            {% endif %}
            </span>
          </div>

          <div class="risk-meter">
            <div class="risk-meter-label">Risk level</div>
            <div class="risk-meter-bar">
              <div class="risk-meter-fill"
                style="width: {% if score is not none %}{{ (score * 100) | round(0) }}{% else %}0{% endif %}%;">
              </div>
            </div>
          </div>

          <div class="risk-line
            {% if score is not none and score < 0.33 %}risk-low{% elif score is not none and score < 0.66 %}risk-medium{% elif score is not none %}risk-high{% endif %}
          ">
            Risk:
            <span>
            {% if score is none %}
              N/A
            {% elif score < 0.33 %}
              LOW
            {% elif score < 0.66 %}
              MEDIUM
            {% else %}
              HIGH
            {% endif %}
            </span>
          </div>
        {% endif %}
      {% endif %}
      <small class="helper">Audio feature–based scoring for uploaded voice samples.</small>
    </div>

  </div>
</div>

<script>
  (function () {
    const overlay = document.getElementById("loadingOverlay");
    const subtitle = document.getElementById("loadingSubtitle");

    function showLoading(text) {
      if (text) subtitle.textContent = text;
      document.body.classList.add("is-loading");
      overlay.classList.add("active");
    }

    document.querySelectorAll("form").forEach(form => {
      form.addEventListener("submit", function () {
        const module = form.querySelector('input[name="module"]')?.value || "";
        const map = {
          "image-text": "Checking for manipulated text…",
          "video": "Scanning video frames…",
          "face-image": "Analyzing face integrity…",
          "voice": "Analyzing audio for cloning patterns…"
        };
        showLoading(map[module] || "Analyzing your file…");
      });
    });

    // Always hide loader once page has rendered (new response)
    window.addEventListener("load", function () {
      document.body.classList.remove("is-loading");
      overlay.classList.remove("active");
    });
  })();
</script>


</body>
</html>

"""


@app.route("/demo", methods=["GET", "POST"])
def demo_dashboard():
    image_text_res = None
    video_res = None
    face_res = None
    voice_res = None

    if request.method == "POST":
        module = request.form.get("module")
        file = request.files.get("file")

        if not file or file.filename == "":
            # Simple error object
            res = {"success": False, "error": "No file uploaded."}
        else:
            path, ext = save_upload(file)
            ext = ext.lower()

            try:
                if module == "image-text":
                    if ext in ALLOWED_IMAGE_EXT:
                        file_type = "image"
                    elif ext in ALLOWED_DOC_EXT:
                        file_type = "document"
                    else:
                        res = {"success": False, "error": f"Unsupported extension: .{ext}"}
                    if ext in ALLOWED_IMAGE_EXT or ext in ALLOWED_DOC_EXT:
                        res = model_image_text(path, file_type)

                    image_text_res = res

                elif module == "video":
                    if ext not in ALLOWED_VIDEO_EXT:
                        res = {"success": False, "error": f"Unsupported video extension: .{ext}"}
                    else:
                        res = model_video_deepfake(path)
                    video_res = res

                elif module == "face-image":
                    if ext not in ALLOWED_IMAGE_EXT:
                        res = {"success": False, "error": f"Unsupported image extension: .{ext}"}
                    else:
                        res = model_face_image(path)
                    face_res = res

                elif module == "voice":
                    if ext not in ALLOWED_AUDIO_EXT:
                        res = {"success": False, "error": f"Unsupported audio extension: .{ext}"}
                    else:
                        res = model_voice_clone(path)
                    voice_res = res
            finally:
                if os.path.exists(path):
                    os.remove(path)

    return render_template_string(
        DEMO_HTML,
        image_text=image_text_res,
        video=video_res,
        face_image=face_res,
        voice=voice_res
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
