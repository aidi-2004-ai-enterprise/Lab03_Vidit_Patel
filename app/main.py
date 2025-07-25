
# Importing necessary libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import xgboost as xgb
import pandas as pd
import numpy as np
import json
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# Enums for island and sex
class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"

class Sex(str, Enum):
    Male = "male"
    Female = "female"

# Pydantic model
class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: Sex
    island: Island

# Initialize FastAPI
app = FastAPI()

# Load the XGBoost model
model_path = os.path.join(os.path.dirname(__file__), "data", "model.json")
model = xgb.XGBClassifier()
try:
    model.load_model(model_path)
    logging.info("Model loaded successfully from %s", model_path)
except Exception as e:
    logging.error("Failed to load model: %s", str(e))
    raise

# Define /predict endpoint
@app.post("/predict")
def predict_penguin(features: PenguinFeatures):
    logging.info("Received prediction request with input: %s", features.dict())

    try:
        # Convert incoming data to DataFrame
        input_data = pd.DataFrame([features.dict()])

        # One-hot encoding
        input_data = pd.get_dummies(input_data)

        # All expected columns from training
        expected_cols = [
            'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
            'body_mass_g', 'year',
            'sex_female', 'sex_male',
            'island_Biscoe', 'island_Dream', 'island_Torgersen'
        ]

        for col in expected_cols:
            if col not in input_data.columns:
                input_data[col] = 0  # fill missing columns

        input_data = input_data[expected_cols]

        # Predict
        prediction = model.predict(input_data)[0]
        species_map = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}
        species = species_map.get(prediction, "Unknown")

        logging.info("Prediction successful: %s", species)

        return {"predicted_species": species}
    
    except Exception as e:
        logging.debug("Prediction failed: %s", str(e))
        raise HTTPException(status_code=400, detail="Prediction failed. Check input values.")
