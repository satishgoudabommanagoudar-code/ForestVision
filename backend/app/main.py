from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .predictor import predict_forest
import shutil
import os

app = FastAPI(
    title="ForestVision API",
    version="1.0.0"
)

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "ForestVision API Running 🌲"
    }


@app.get("/health")
def health():
    return {
        "status": "Running"
    }


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_forest(filepath)

    return result


# Keep this route so your current frontend still works
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return await classify(file)