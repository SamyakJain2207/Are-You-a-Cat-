from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import tempfile

# Import existing inference logic
from pipelines.inference import load_model, predict

app = FastAPI(title="Cat vs Not Cat API")

# Load model once at startup
model = load_model()


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name

        # Run prediction
        result = predict(temp_path, model)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )