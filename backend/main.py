from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from model import load_model, predict_image
from PIL import Image
import io

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://deepfake-detector-theta.vercel.app/"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()

@app.get("/")
async def root():
    return {"message": "Welcome to the Deepfake Detection API. Use the /predict endpoint to analyze images."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    prediction = predict_image(model, image)
    return prediction