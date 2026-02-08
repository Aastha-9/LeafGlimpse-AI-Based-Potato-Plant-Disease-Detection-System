from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

from translations import translations
from recommendations import recommendations

app = FastAPI()

# ========================
# CORS
# ========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Class names (same order as training)
# ========================
class_names = [
    "Potato___Late_blight",
    "Potato___Early_blight",
    "Potato___healthy"
]

# ========================
# Load latest SavedModel
# ========================
MODEL_DIR = "../models"
latest_version = max([int(i) for i in os.listdir(MODEL_DIR) if i.isdigit()])

model = tf.keras.layers.TFSMLayer(
    f"{MODEL_DIR}/{latest_version}",
    call_endpoint="serve"
)

# ========================
# Routes
# ========================
@app.get("/")
def home():
    return {"message": "Potato Disease Classification API"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    lang: str = Query("en")   # en | hi | mr
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((256, 256))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model(img_array)

    index = int(np.argmax(predictions))
    confidence = float(np.max(predictions)) * 100
    predicted_class = class_names[index]

    return {
        "disease": predicted_class,
        "confidence": round(confidence, 2),
        "message": translations[predicted_class][lang],
        "recommendations": recommendations[predicted_class][lang]
    }
