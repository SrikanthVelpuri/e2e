# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import uvicorn
from typing import List, Dict
import numpy as np
import logging
from prometheus_client import make_asgi_app, Counter, Histogram
from contextlib import asynccontextmanager
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total number of predictions made')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Time spent processing prediction')

class PredictionInput(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

# Model loading context
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model on startup
    try:
        # Get the latest version of the model from MLflow
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions("adult_income_classifier", stages=["Production"])[0]
        app.state.model = mlflow.pyfunc.load_model(f"models:/adult_income_classifier/{latest_version.version}")
        app.state.model_version = latest_version.version
        logger.info(f"Loaded model version: {latest_version.version}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    yield
    # Cleanup on shutdown
    app.state.model = None

app = FastAPI(lifespan=lifespan)

# Add prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    try:
        with PREDICTION_LATENCY.time():
            # Convert input to DataFrame
            df = pd.DataFrame([input_data.features])
            
            # Make prediction
            prediction_proba = app.state.model.predict_proba(df)
            prediction = app.state.model.predict(df)[0]
            
            # Increment prediction counter
            PREDICTION_COUNTER.inc()
            
            return PredictionResponse(
                prediction=int(prediction),
                probability=float(prediction_proba[0][1]),
                model_version=app.state.model_version
            )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_version": app.state.model_version}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)