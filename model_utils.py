# model_utils.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib

# Load and preprocess dataset
def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    le = LabelEncoder()
    df['Crop_Type'] = le.fit_transform(df['Crop_Type'])
    return df, le

# Train models and save
def train_models(df):
    X = df.drop(columns=["Farm_ID", "Crop_Yield_ton", "Sustainability_Score"])
    y_yield = df["Crop_Yield_ton"]
    y_sustain = df["Sustainability_Score"]

    X_train, X_test, y_yield_train, y_yield_test = train_test_split(X, y_yield, test_size=0.2, random_state=42)
    _, _, y_sustain_train, y_sustain_test = train_test_split(X, y_sustain, test_size=0.2, random_state=42)

    yield_model = RandomForestRegressor(n_estimators=100, random_state=42)
    sustain_model = RandomForestRegressor(n_estimators=100, random_state=42)

    yield_model.fit(X_train, y_yield_train)
    sustain_model.fit(X_train, y_sustain_train)

    # Save models
    joblib.dump(yield_model, "yield_model.pkl")
    joblib.dump(sustain_model, "sustainability_model.pkl")

    # Evaluate
    y_pred_yield = yield_model.predict(X_test)
    y_pred_sustain = sustain_model.predict(X_test)

    print("Yield RMSE:", np.sqrt(mean_squared_error(y_yield_test, y_pred_yield)))
    print("Sustainability RMSE:", np.sqrt(mean_squared_error(y_sustain_test, y_pred_sustain)))

# Predict from user input
def predict(input_data, yield_model, sustain_model):
    yield_pred = yield_model.predict(input_data)
    sustain_pred = sustain_model.predict(input_data)
    return yield_pred[0], sustain_pred[0]
