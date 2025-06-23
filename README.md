# PREDICTION OF WEATHER CONDITIONS BASED ON SCREEN IMAGES USING CNN

## Overview
This project is focused on building a weather prediction system based on sky images using Convolutional Neural Networks (CNN). The model is capable of detecting and classifying weather conditions such as sunny, cloudy, and rainy from static sky images. The system is trained using the "Weather Recognizer with CNN" dataset from Kaggle.

### Project Details
- **Name:** Wiwin Sigalingging
- **NIM:** 2155301162
- **Class:** 4 TI B
- **Instructor:** Ananda, S.Kom., M.T., Ph.D.
- **Team Member:** Muhammad Anwar, S.Tr.Kom
- **Institution:** Politeknik Caltex Riau
- **Academic Year:** 2024/2025

## Dataset
The dataset used in this project is the [Weather Recognizer with CNN dataset](https://www.kaggle.com/datasets/abhay06102003/weather-recognizer-with-cnn) from Kaggle. It consists of sky images classified into three weather conditions:
- Sunny
- Cloudy
- Rainy

## Methodology
### Data Preprocessing
- Images are resized to 150x150 pixels.
- Data augmentation techniques such as rotation, flipping, zoom, and shift are applied to increase data variability.

### CNN Architecture
- The model consists of multiple layers:
  - **Convolutional Layers (Conv2D)** for feature extraction (edge, texture, and patterns).
  - **MaxPooling Layers (MaxPooling2D)** for dimensionality reduction.
  - **Flatten Layer** to convert 2D features to a 1D vector.
  - **Dense Layers** for classification.
  - **Dropout** layer to prevent overfitting.

### Model Training
- Model is trained using the **Adam optimizer** and **categorical cross-entropy loss**.
- Early stopping is used to prevent overfitting during training, with a maximum of 30 epochs.

### Evaluation Metrics
- The model's performance is evaluated using accuracy, loss, precision, recall, and F1-score.

## Results
- **Training Accuracy:** 87.95% with a loss of 0.2886.
- **Validation Accuracy:** 92.21% with a loss of 0.2831.
The model shows great performance with high accuracy and low loss values, demonstrating its ability to generalize well to new sky images.

## Deployment
The trained model is saved as `weather_model.h5` and deployed using **Streamlit** to create a web application where users can upload sky images to predict weather conditions.

### Streamlit Application
1. **Upload Image:** Users can upload sky images in JPG, JPEG, or PNG format.
2. **Weather Prediction:** The model classifies the image and predicts the weather condition (sunny, cloudy, or rainy).
The web interface displays the predicted class and confidence level for the uploaded image.

## Installation
### Prerequisites
- Python 3.x
- TensorFlow
- Streamlit
- Other required Python libraries (e.g., NumPy, Pandas)

### Steps to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/weather-prediction-cnn.git
   cd weather-prediction-cnn
