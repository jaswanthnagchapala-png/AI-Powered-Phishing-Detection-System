
from pathlib import Path
import sys

from flask import Flask, render_template, request
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from features import extract_features

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static")
)

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

saved_data = joblib.load(MODEL_PATH)
model = saved_data["model"]
feature_names = saved_data["feature_names"]


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    submitted_url = ""
    error = None

    if request.method == "POST":
        submitted_url = request.form.get("url", "").strip()

        if not submitted_url:
            error = "Please enter a URL."
        elif len(submitted_url) > 2048:
            error = "The URL is too long."
        else:
            try:
                features = extract_features(submitted_url)
                feature_values = [
                    features[name] for name in feature_names
                ]

                prediction = model.predict(
                    pd.DataFrame([feature_values], columns=feature_names)
                )[0]

                if int(prediction) == 1:
                    result = "Phishing"
                else:
                    result = "Legitimate"

            except Exception:
                error = "Unable to analyze this URL. Please check it and try again."

    return render_template(
        "index.html",
        result=result,
        submitted_url=submitted_url,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
