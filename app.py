import logging

from flask import Flask, render_template, request, jsonify

from translate import translate, warm_up

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MAX_INPUT_CHARS = 500


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Please enter some text to translate."}), 400

    if len(text) > MAX_INPUT_CHARS:
        return jsonify({"error": f"Text is too long (max {MAX_INPUT_CHARS} characters)."}), 400

    try:
        translation = translate(text)
    except Exception:
        logger.exception("Translation failed")
        return jsonify({"error": "Translation failed. Please try again."}), 500

    return jsonify({"translation": translation})


if __name__ == "__main__":
    logger.info("Loading translation model...")
    warm_up()
    logger.info("Model loaded. Starting server.")
    app.run(debug=True)
