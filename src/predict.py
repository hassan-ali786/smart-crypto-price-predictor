import joblib
import pandas as pd


FEATURES = [
    "Open",
    "High",
    "Low",
    "Volume",
    "MA7",
    "MA14",
    "Lag1",
    "Lag2",
    "Lag3",
    "Returns",
    "RSI"
]


def load_model(path="model.pkl"):
    model = joblib.load(path)
    return model


def predict_price(model, input_data):
    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )

    prediction = model.predict(input_df)

    return round(prediction[0], 2)


if __name__ == "__main__":

    model = load_model()

    sample_input = [
        65000,
        65500,
        64500,
        5000000,
        64800,
        64000,
        64900,
        64700,
        64600,
        0.01,
        58
    ]

    prediction = predict_price(
        model,
        sample_input
    )

    print("Predicted Close Price:", prediction)
