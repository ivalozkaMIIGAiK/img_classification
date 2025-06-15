from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
import uvicorn
import cv2
from PIL import Image
import io
import joblib

app = FastAPI()

# Загрузка модели и LabelBinarizer
model = load_model("best_model_final.h5", compile=False)
lb = joblib.load("label_binarizer.pkl")

# Предобработка изображения
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # (1, 64, 64, 3)
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = preprocess_image(contents)

        predictions = model.predict(image)[0]
        predicted_index = np.argmax(predictions)
        predicted_class = lb.classes_[predicted_index]

        probabilities = {
            lb.classes_[i]: float(pred)
            for i, pred in enumerate(predictions)
        }

        return JSONResponse(content={
            "predicted_class": predicted_class,
            "probabilities": probabilities
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

