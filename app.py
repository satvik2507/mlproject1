from flask import Flask, render_template, request
from src.MLproject.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    message = request.form["message"]

    pipeline = PredictPipeline()

    prediction = pipeline.predict(message)

    if prediction == 1:
        result = "Spam"
    else:
        result = "Ham"

    return render_template(
        "index.html",
        prediction=result,
        message=message
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False
    )