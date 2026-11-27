"""
NOOL - AI Studio + Image Enhancer backend

Endpoints:
  POST /api/generate    -> AI Design Studio (text/specs -> design image)
  POST /api/enhance     -> AI Image Enhancer (rough photo -> cleaned-up image)
  POST /api/weave-draft -> Loom-ready weave draft + feasibility score
  GET  /api/file/<fn>   -> Serve generated/enhanced images
  GET  /health          -> Simple healthcheck
  GET  /                -> Frontend web UI
"""

import os
import uuid
from dotenv import load_dotenv

# Load variables from .env file before anything else
load_dotenv()

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from modules.enhancer import ImageEnhancer
from modules.design_generator import DesignGenerator
from modules.weave_draft import generate_weave_draft

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB upload cap

enhancer = ImageEnhancer(dnn_model_path=os.environ.get("SR_MODEL_PATH"))
design_generator = DesignGenerator()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """
    Body (JSON or form):
      garment_type, fabric, colour, embroidery, style_notes, custom_prompt
    Returns: { prompt, image_url, source }
    """
    data = request.form if request.form else (request.get_json(silent=True) or {})

    try:
        result = design_generator.generate(
            garment_type=data.get("garment_type", ""),
            fabric=data.get("fabric", ""),
            colour=data.get("colour", ""),
            embroidery=data.get("embroidery", ""),
            style_notes=data.get("style_notes", ""),
            custom_prompt=data.get("custom_prompt", ""),
        )

        filename = f"design_{uuid.uuid4().hex[:10]}.png"
        output_path = os.path.join(OUTPUT_DIR, filename)
        with open(output_path, "wb") as f:
            f.write(result["image_bytes"])

        return jsonify({
            "prompt": result["prompt"],
            "source": result["source"],
            "image_url": f"/api/file/{filename}",
        })
    except Exception as e:
        print(f"[Generate Error] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/enhance", methods=["POST"])
def api_enhance():
    """
    Multipart form:
      image: file
      scale: "2" or "4" (optional, default 2)
    Returns: { image_url }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided (field name must be 'image')"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid or missing file. Allowed: png, jpg, jpeg, webp"}), 400

    scale = int(request.form.get("scale", 2))
    scale = 4 if scale >= 4 else 2

    input_filename = f"in_{uuid.uuid4().hex[:10]}_{secure_filename(file.filename)}"
    input_path = os.path.join(UPLOAD_DIR, input_filename)
    file.save(input_path)

    output_filename = f"enhanced_{uuid.uuid4().hex[:10]}.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    try:
        enhancer.enhance(input_path, output_path, scale=scale)
    except Exception as e:
        print(f"[Enhance Error] {e}")
        return jsonify({"error": f"Enhancement failed: {e}"}), 500

    return jsonify({"image_url": f"/api/file/{output_filename}"})


@app.route("/api/weave-draft", methods=["POST"])
def api_weave_draft():
    """
    Turns a design image into a loom-ready graph-paper draft + feasibility score.

    Either:
      - form field 'source_file': filename of an image already in outputs/
      - multipart file field 'image': a fresh upload
    """
    source_file = request.form.get("source_file")
    image_path = None

    if source_file:
        candidate = os.path.join(OUTPUT_DIR, secure_filename(source_file))
        if not os.path.isfile(candidate):
            return jsonify({"error": "source_file not found"}), 400
        image_path = candidate
    elif "image" in request.files and request.files["image"].filename:
        file = request.files["image"]
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg, webp"}), 400
        input_filename = f"draftsrc_{uuid.uuid4().hex[:10]}_{secure_filename(file.filename)}"
        image_path = os.path.join(UPLOAD_DIR, input_filename)
        file.save(image_path)
    else:
        return jsonify({"error": "Provide 'source_file' (existing output) or an 'image' upload"}), 400

    def _int_form(key, default, lo, hi):
        try:
            return max(lo, min(hi, int(request.form.get(key, default))))
        except (TypeError, ValueError):
            return default

    grid_cols = _int_form("grid_cols", 48, 16, 96)
    grid_rows = _int_form("grid_rows", 64, 16, 128)
    n_colors = _int_form("n_colors", 6, 2, 12)

    try:
        result = generate_weave_draft(
            image_path,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            n_colors=n_colors
        )
    except Exception as e:
        print(f"[Weave Draft Error] {e}")
        return jsonify({"error": f"Weave draft generation failed: {e}"}), 500

    filename = f"weavedraft_{uuid.uuid4().hex[:10]}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, "wb") as f:
        f.write(result["draft_image_bytes"])

    return jsonify({
        "image_url": f"/api/file/{filename}",
        "score": result["score"],
        "score_label": result["score_label"],
        "breakdown": result["breakdown"],
        "palette": result["palette"],
        "grid": result["grid"],
        "disclaimer": result["disclaimer"],
    })


@app.route("/api/file/<filename>")
def get_file(filename):
    return send_from_directory(OUTPUT_DIR, secure_filename(filename))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug=False prevents Windows watchdog process duplication loops
    app.run(host="127.0.0.1", port=port, debug=False)