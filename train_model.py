# train_model.py

from model_utils import load_and_preprocess_data, train_models

df, _ = load_and_preprocess_data("farmer_advisor_dataset.csv")
train_models(df)
