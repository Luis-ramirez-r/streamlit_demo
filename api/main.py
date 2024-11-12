# main.py
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import tomli
from pathlib import Path
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Engine Maintenance Risk Predictor API")

class PredictionRequest(BaseModel):
    sensor_values: Dict[str, float]

class PredictionResponse(BaseModel):
    risk_level: int
    risk_probabilities: List[float]
    risk_label: str
    color: str

# Global variables
model = None
scaler = None
feature_order = None

def load_config():
    """Load configuration from TOML file."""
    try:
        with open('config.toml', 'rb') as f:
            return tomli.load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise RuntimeError(f"Failed to load config: {e}")

def get_risk_label(risk_class: int) -> tuple:
    """Convert risk class to descriptive label with color."""
    risk_labels = {
        0: ("Low Risk (>30 cycles)", "green"),
        1: ("Medium Risk (20-30 cycles)", "yellow"),
        2: ("High Risk (10-20 cycles)", "orange"),
        3: ("Higher Risk (0-10 cycles)", "red")
    }
    return risk_labels.get(risk_class, ("Unknown", "gray"))

@app.on_event("startup")
async def load_model_artifacts():
    """Load model artifacts on startup."""
    global model, scaler, feature_order
    
    config = load_config()
    model_dir = Path(config['paths']['models']) / config['model_config']['maintenance_classification_dir']
    
    logger.info(f"Loading model artifacts from {model_dir}")
    
    try:
        model_path = model_dir / 'logistic_model.joblib'
        scaler_path = model_dir / 'feature_scaler.joblib'
        
        if not model_path.exists() or not scaler_path.exists():
            raise RuntimeError("Model files not found")
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Get feature order from config
        feature_order = (config['sensor_groups']['group1'] + 
                        config['sensor_groups']['group2'])
        
        logger.info(f"Loaded feature order: {feature_order}")
        logger.info("Successfully loaded model artifacts")
        
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {str(e)}")
        raise RuntimeError(f"Failed to load model artifacts: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction based on sensor values."""
    try:
        # Create DataFrame with explicit feature names
        input_data = {name: [request.sensor_values[name]] for name in feature_order}
        input_df = pd.DataFrame(input_data, columns=feature_order)
        
        # Scale the input
        scaled_input = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0].tolist()
        
        # Get risk label and color
        risk_label, color = get_risk_label(prediction)
        
        return PredictionResponse(
            risk_level=int(prediction),
            risk_probabilities=prediction_proba,
            risk_label=risk_label,
            color=color
        )
    except KeyError as e:
        logger.error(f"Missing feature in input: {e}")
        raise HTTPException(status_code=400, detail=f"Missing feature in input: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "feature_order": feature_order}