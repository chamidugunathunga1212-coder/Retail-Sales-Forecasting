from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.pipelines.prediction_pipeline import PredictionPipeline
from src.entity.config_entity.prediction_input import PredictionInputConfig

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import io

app = FastAPI(title="Walmart Sales Forecast API", version="1.0.0")

PREPROCESSOR_PATH = PredictionInputConfig.preprocessor_stable_path
MODEL_PATH = PredictionInputConfig.model_stable_path
FEATURE_CFG_PATH = PredictionInputConfig.feature_configure_file_path

pipeline = PredictionPipeline(
    preprocessor_path=PREPROCESSOR_PATH,
    model_path=MODEL_PATH,
    feature_config_path=FEATURE_CFG_PATH
)

REQUIRED_COLUMNS = [
    "Store", "Dept", "Date", "IsHoliday", "Temperature", "Fuel_Price",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "CPI", "Unemployment", "Type", "Size"
]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload a CSV with feature columns.
    Returns a CSV with an added column: Weekly_Sales_Prediction
    """

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")
    

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {str(e)}")
    


    # Validate required columns (allow extra columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing}"
        )
    
    try:
        preds = pipeline.predict(df[REQUIRED_COLUMNS])
        df_out = df.copy()
        df_out["Weekly_Sales_Prediction"] = preds
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    

    # Convert output dataframe to CSV bytes
    buffer = io.StringIO()
    df_out.to_csv(buffer, index=False)
    buffer.seek(0)


    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predictions_{file.filename}"}
    )