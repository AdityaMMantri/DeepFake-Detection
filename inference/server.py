"""
server.py  —  Flask web server for the DeepFake Detector HTML UI.

Imports load_model and predict EXACTLY the same way app.py does.
app.py is not modified at all — run this file for the HTML dashboard.

Usage:
    python server.py
    then open http://localhost:5000
"""

import os
from flask import Flask, request, jsonify, render_template
from PIL import Image

# ── Same imports as app.py ────────────────────────────────
from load_model import load_model
from predict import predict

# ── Same MODEL_PATHS dict as app.py ──────────────────────
MODEL_PATHS = {
    "cnn":        r"F:\SEM-6\DL\DEEP-FAKE\DL-Project\checkpoints\cnn\best_model.pth",
    "vit_9ch":    r"F:\SEM-6\DL\DEEP-FAKE\DL-Project\checkpoints\vit\best_model.pth",
    "vit_single": r"F:\SEM-6\DL\DEEP-FAKE\DL-Project\checkpoints\vit_single\best_auc.pth",
}

app = Flask(__name__)

# Simple dict cache — mirrors st.cache_resource behaviour
_model_cache: dict = {}


def get_model(model_type: str):
    """Load and cache model, same as @st.cache_resource get_model() in app.py."""
    if model_type not in _model_cache:
        print(f"[LOADING MODEL]: {model_type}")   # same print as app.py
        _model_cache[model_type] = load_model(model_type)
    return _model_cache[model_type]


# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/model-info")
def model_info():
    model_type = request.args.get("model_type", "cnn")
    if model_type not in MODEL_PATHS:
        return jsonify({"error": "Unknown model type"}), 400

    full_path = os.path.abspath(MODEL_PATHS[model_type])

    # Mirrors the debug block in app.py exactly
    active_class = type(_model_cache[model_type]).__name__ \
                   if model_type in _model_cache else "Not loaded yet"

    return jsonify({
        "selected_model":  model_type,
        "checkpoint_path": full_path,
        "exists":          os.path.exists(full_path),
        "active_class":    active_class,
    })


@app.route("/reload", methods=["POST"])
def reload_model():
    """Mirrors the st.cache_resource.clear() button in app.py."""
    _model_cache.clear()
    return jsonify({"status": "ok", "message": "Cache cleared. Model will reload."})


@app.route("/predict", methods=["POST"])
def run_predict():
    model_type = request.form.get("model_type", "cnn")
    if model_type not in MODEL_PATHS:
        return jsonify({"error": "Unknown model type"}), 400

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400

    # ── Exactly what app.py does ──────────────────────────
    image  = Image.open(file.stream).convert("RGB")   # same as app.py
    model  = get_model(model_type)                     # same as app.py
    result = predict(model, image, model_type)         # same as app.py
    # ─────────────────────────────────────────────────────

    # Serialise to JSON (convert numpy floats safely)
    def f(v):
        return float(v) if v is not None else None

    response = {
        "prediction": result["prediction"],
        "confidence": f(result["confidence"]),
        "p_fake":     f(result["p_fake"]),
    }

    if result.get("extra"):
        extra = result["extra"]
        response["extra"] = {
            "p_vit":       f(extra.get("p_vit")),
            "trust_score": f(extra.get("trust_score")),
            "explanation": extra.get("explanation"),
        }

    return jsonify(response)


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)