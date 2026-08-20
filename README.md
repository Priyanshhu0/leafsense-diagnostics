# 🌿 LeafSense Diagnostics

### AI-Powered Plant Disease Detection & Classification System

**LeafSense Diagnostics** is a deep-learning-based web application designed to detect and classify **tomato leaf diseases from uploaded images**.

The system combines a **Convolutional Neural Network (CNN)** with a **FastAPI backend** and a lightweight web frontend to provide automated plant disease predictions and confidence scores.

##  Overview

Plant diseases can significantly affect crop productivity and agricultural output. Early identification of symptoms can help in taking appropriate action before the disease spreads.

LeafSense Diagnostics provides a simple AI-based approach:

```text
Upload Leaf Image
       ↓
Image Preprocessing
       ↓
CNN-Based Classification
       ↓
Disease Prediction
       ↓
Confidence Score
```

The current model focuses on **four tomato leaf classes**.

---

##  Key Features

- 🌱 Tomato leaf disease detection
- 🧠 CNN-based image classification
- 📷 Single-image prediction
- ⚡ FastAPI inference backend
- 📊 Prediction confidence score
  
---

##  Supported Classes

The current model recognizes four tomato leaf conditions:

| Class | Category |
|---|---|
| 🟢 Tomato Healthy | Healthy |
| 🦠 Tomato Bacterial Spot | Bacterial Disease |
| 🍂 Tomato Early Blight | Fungal Disease |
| 🍁 Tomato Late Blight | Fungal Disease |

---

#  How It Works

### 1. Image Upload

The user uploads a photograph of a tomato leaf through the web interface.

### 2. Image Preprocessing

The uploaded image is processed by the backend.

The image is:

- Resized to **128 × 128 pixels**
- Pixel values are scaled
- Converted into a format suitable for model inference

### 3. CNN Inference

The processed image is passed to the trained CNN model.

### 4. Classification

The model predicts the most likely class from the four supported categories.

### 5. Result

The application returns:

```text
Predicted Disease
        +
Confidence Percentage
```
---

#  Technology Stack

## Frontend

- HTML5
- CSS3
- JavaScript

## Backend

- Python
- FastAPI
- Uvicorn

## Machine Learning

- TensorFlow
- Keras
- Convolutional Neural Network

## Image Processing

- Pillow

---

#  Project Structure

```text
leafsense-diagnostics/
│
├── backend/
│   ├── main.py
│   ├── predict.py
│   ├── model_loader.py
│   ├── utils.py
│   ├── requirements.txt
│   │
│   └── model/
│       └── plant_disease_model.keras
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── training/
│   └── train_model.py
│
└── README.md
```

---

#  Installation

## Prerequisites

Make sure you have installed:

- Python 3.x
- pip
- Git

---

## 1. Clone the Repository

```bash
git clone https://github.com/Priyanshhu0/leafsense-diagnostics.git
```

Enter the project directory:

```bash
cd leafsense-diagnostics
```

---

#  2. Create Virtual Environment

Navigate to the backend:

```bash
cd backend
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

#  3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

#  Running the Backend

From the `backend` directory:

```bash
uvicorn main:app --reload
```

The FastAPI server will start at:

```text
http://127.0.0.1:8000
```

---

#  Running the Frontend

Open another terminal.

Navigate to the frontend:

```bash
cd frontend
```

Start a local server:

```bash
python -m http.server 5500
```

Open the application:

```text
http://localhost:5500
```

Make sure the backend is running before making predictions.

---

#  Model Training

The project includes a dedicated training pipeline.

Navigate to the training directory:

```bash
cd training
```

Run:

```bash
python train_model.py
```

The training pipeline:

```text
Dataset
   ↓
Image Loading
   ↓
Preprocessing
   ↓
CNN Training
   ↓
Validation
   ↓
Best Model
   ↓
plant_disease_model.keras
```

The trained model is used by the FastAPI backend for real-time image prediction.

---

#  Dataset

The project is designed around the **PlantVillage dataset**, using the tomato subset.

The model currently uses these four classes:

```text
Tomato___healthy
Tomato___Bacterial_spot
Tomato___Early_blight
Tomato___Late_blight
```

Recommended dataset structure:

```text
dataset/
│
├── train/
│   ├── Tomato___healthy/
│   ├── Tomato___Bacterial_spot/
│   ├── Tomato___Early_blight/
│   └── Tomato___Late_blight/
│
└── val/
    ├── Tomato___healthy/
    ├── Tomato___Bacterial_spot/
    ├── Tomato___Early_blight/
    └── Tomato___Late_blight/
```


