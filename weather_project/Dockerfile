# Gunakan image python ringan
FROM python:3.10-slim

# Set direktori kerja dalam container
WORKDIR /app

# Copy semua isi folder ke dalam container
COPY . .

# Install pip & dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Port yang akan dibuka Hugging Face (wajib 7860 atau 8080)
EXPOSE 7860

# Jalankan FastAPI app menggunakan uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
