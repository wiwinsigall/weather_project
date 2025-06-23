from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename

# Inisialisasi Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Pastikan folder upload ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
model = tf.keras.models.load_model("weather_model.h5")

# Load nama kelas
with open("classes.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f]

# Halaman utama
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files['file']
        if file:
            # Simpan file upload
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Proses gambar
            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Prediksi
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            predicted_label = class_names[predicted_index]
            confidence = float(prediction[0][predicted_index]) * 100

            # Tampilkan hasil di template
            return render_template("index.html",
                                   image_file=filename,
                                   label=predicted_label,
                                   confidence=confidence)

    return render_template("index.html")

# Jalankan Flask
if __name__ == "__main__":
    app.run(debug=True)
