from fastapi import FastAPI, UploadFile, File
import shutil, os
from backend.inference import predict_alzheimer

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
@app.get("/")
def home():
    return {"message": "Alzheimer API is running. Go to /docs"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_alzheimer(file_path)
    return result
