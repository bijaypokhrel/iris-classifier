import joblib
import pandas as pd
import numpy as np
from data_loader import load_iris_df

def predict_sample(features):
    # features: iterable of four numbers matching iris feature order
    model = joblib.load("models/rf_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    arr = np.array(features).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)[0]
    # get human-readable name
    df = load_iris_df()
    target_names = df['target_name'].unique()
    return pred, target_names[pred]

if __name__ == "__main__":
    sample = [5.1, 3.5, 1.4, 0.2]
    label, name = predict_sample(sample)
    print("Predicted label:", label, "->", name)
