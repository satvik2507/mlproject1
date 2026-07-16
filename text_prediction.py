from src.MLproject.pipeline.predict_pipeline import PredictPipeline

pipeline = PredictPipeline()

messages = [
    "Congratulations! You have won a free iPhone. Call now to claim your prize.",
    "Hey, are we still meeting tomorrow at 10 AM?",
    "URGENT! You have been selected for a cash reward. Click the link now.",
    "Can you send me the lecture notes?",

    "congratulations! You have won my free only fans subscription . Click the link to claim your prize .",
]

for message in messages:
    prediction = pipeline.predict(message)

    if prediction == 1:
        label = "SPAM"
    else:
        label = "HAM"

    print("-" * 60)
    print("Message:", message)
    print("Prediction:", label)
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder

# df = pd.read_csv("artifacts/train.csv")

# le = LabelEncoder()

# df["Category"] = le.fit_transform(df["Category"])

# print(le.classes_)
# import pickle

# with open("artifacts/model.pkl", "rb") as f:
#     model = pickle.load(f)

# print(model)