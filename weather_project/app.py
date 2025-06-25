from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import shutil
import os
import base64

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Mount static folder (untuk CSS, JS, dsb jika perlu)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model dan kelas
model = tf.keras.models.load_model("weather_model.h5")
with open("classes.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f]

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, file: UploadFile = File(...)):
    filename = file.filename
    filepath = os.path.join("/tmp", filename)  # simpan di /tmp agar tidak kena permission error

    # Simpan file upload
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Proses gambar
    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = float(prediction[0][predicted_index]) * 100

    # Encode gambar ke base64
    with open(filepath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        mime_type = file.content_type or "image/jpeg"
        image_data_uri = f"data:{mime_type};base64,{encoded_string}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "image_data": image_data_uri,
        "label": predicted_label,
        "confidence": confidence
    })
