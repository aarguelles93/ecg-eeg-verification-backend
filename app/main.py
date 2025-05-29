from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from app.inference import predict_signal
from app.utils import validate_and_preprocess
from app.model_config import MODEL_CONFIG

from fastapi import Request

app = FastAPI()

# In production, restrict to allow_origins only frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AVAILABLE_MODELS = ["tcn", "cnn", "cnn_lstm", "mlp", "svm"]

@app.get("/")
def read_root():
    return {"message": "ECG/EEG Verification API is running."}

@app.get("/available-models")
def get_models():
    return {"available_models": list(MODEL_CONFIG.keys())}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query("tcn", enum=["tcn", "cnn", "cnn_lstm", "mlp", "svm"])
):
    print(f"[DEBUG] Received content_type: {file.content_type}")

    ACCEPTED_CSV_TYPES = ["text/csv", "application/vnd.ms-excel", "application/octet-stream"]
    if file.content_type not in ACCEPTED_CSV_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Only CSV files are supported. Received: {file.content_type}"
        )

    df = pd.read_csv(file.file)
    try:
        data = validate_and_preprocess(df)
        result = predict_signal(data, model_name=model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
