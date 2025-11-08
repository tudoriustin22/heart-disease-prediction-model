from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
from typing import Literal
import os

# Initialize FastAPI API server for serving the model
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predict heart disease risk using Machine Learning",
    version="1.0.0"
)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models when starting up the server
try:
    model_path = os.path.join(BASE_DIR, 'heart_disease_xgb_model.json')  # load from JSON
    preprocessor_path = os.path.join(BASE_DIR, 'preprocessor.pkl')

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_path)

    # Load preprocessor (which handles the one-hot encoding for the data)
    preprocessor = joblib.load(preprocessor_path)

    print("Models loaded successfully!")
    print(f"  - Model: {model_path}")
    print(f"  - Preprocessor: {preprocessor_path}")
except FileNotFoundError as e:
    print(f"Error: Model files not found.")
    raise
except Exception as e:
    print(f"Error loading models: {e}")
    raise


# Define input schema with validation
class PatientData(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Age": 54,
                "Sex": "M",
                "ChestPainType": "NAP",
                "RestingBP": 150,
                "Cholesterol": 195,
                "FastingBS": 0,
                "RestingECG": "Normal",
                "MaxHR": 122,
                "ExerciseAngina": "N",
                "Oldpeak": 0.0,
                "ST_Slope": "Up"
            }
        }
    )

    Age: int = Field(..., ge=1, le=120, description="Patient age in years")
    Sex: Literal['M', 'F'] = Field(..., description="Patient sex (M/F)")
    ChestPainType: Literal['ATA', 'NAP', 'ASY', 'TA'] = Field(..., description="Chest pain type")
    RestingBP: int = Field(..., ge=0, le=300, description="Resting blood pressure (mm Hg)")
    Cholesterol: int = Field(..., ge=0, le=600, description="Serum cholesterol (mg/dl)")
    FastingBS: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)")
    RestingECG: Literal['Normal', 'ST', 'LVH'] = Field(..., description="Resting ECG results")
    MaxHR: int = Field(..., ge=60, le=250, description="Maximum heart rate achieved")
    ExerciseAngina: Literal['Y', 'N'] = Field(..., description="Exercise induced angina (Y/N)")
    Oldpeak: float = Field(..., ge=-5.0, le=10.0, description="ST depression induced by exercise")
    ST_Slope: Literal['Up', 'Flat', 'Down'] = Field(..., description="Slope of peak exercise ST segment")


# Define response schema
class PredictionResponse(BaseModel):
    prediction: int
    has_heart_disease: bool
    risk_level: str
    probability: dict
    confidence: float
    message: str


# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "accuracy": "87.5%",
        "model": "XGBoost with GridSearchCV",
        "endpoints": {
            "predict": "/predict - Single patient prediction",
            "batch": "/predict/batch - Multiple patients",
            "health": "/health - Health check",
            "docs": "/docs - API documentation"
        }
    }


# Health check API endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": xgb_model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_type": type(xgb_model).__name__
    }


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict_heart_disease(patient: PatientData):
    try:
        # Convert input to DataFrame (must match original column order)
        input_data = pd.DataFrame([{
            'Age': patient.Age,
            'Sex': patient.Sex,
            'ChestPainType': patient.ChestPainType,
            'RestingBP': patient.RestingBP,
            'Cholesterol': patient.Cholesterol,
            'FastingBS': patient.FastingBS,
            'RestingECG': patient.RestingECG,
            'MaxHR': patient.MaxHR,
            'ExerciseAngina': patient.ExerciseAngina,
            'Oldpeak': patient.Oldpeak,
            'ST_Slope': patient.ST_Slope
        }])

        # Apply preprocessing (StandardScaler + OneHotEncoder)
        X_processed = preprocessor.transform(input_data)

        # Make prediction
        prediction = xgb_model.predict(X_processed)[0]
        probability = xgb_model.predict_proba(X_processed)[0]

        # Calculate confidence and risk level
        confidence = float(max(probability))
        prob_disease = float(probability[1])

        # Determine risk level
        if prob_disease < 0.3:
            risk_level = "Low"
            message = "Low risk of heart disease. Continue healthy lifestyle."
        elif prob_disease < 0.6:
            risk_level = "Moderate"
            message = "Moderate risk. Consider consulting with a healthcare provider."
        else:
            risk_level = "High"
            message = "High risk detected. Please consult a healthcare professional immediately."

        # Return response
        return PredictionResponse(
            prediction=int(prediction),
            has_heart_disease=bool(prediction),
            risk_level=risk_level,
            probability={
                "no_disease": float(probability[0]),
                "has_disease": float(probability[1])
            },
            confidence=confidence,
            message=message
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Batch prediction endpoint
@app.post("/predict/batch")
def predict_batch(patients: list[PatientData]):
    try:
        results = []
        for idx, patient in enumerate(patients):
            # Convert to DataFrame
            input_data = pd.DataFrame([patient.model_dump()])

            # Preprocess and predict
            X_processed = preprocessor.transform(input_data)
            prediction = xgb_model.predict(X_processed)[0]
            probability = xgb_model.predict_proba(X_processed)[0]

            # Determine risk level
            prob_disease = float(probability[1])
            if prob_disease < 0.3:
                risk_level = "Low"
            elif prob_disease < 0.6:
                risk_level = "Moderate"
            else:
                risk_level = "High"

            results.append({
                "patient_id": idx + 1,
                "prediction": int(prediction),
                "has_heart_disease": bool(prediction),
                "risk_level": risk_level,
                "probability": {
                    "no_disease": float(probability[0]),
                    "has_disease": float(probability[1])
                }
            })

        return {
            "total_patients": len(patients),
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# Run the app
if __name__ == "__main__":
    import uvicorn

    print("Starting Heart Disease Prediction API...")
    print("Model Accuracy: 87.5%")
    print("Test endpoint: http://localhost:8000/predict\n")
    uvicorn.run("deployment_inference:app", host="0.0.0.0", port=8000, reload=True)
