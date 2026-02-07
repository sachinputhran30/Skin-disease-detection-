# Skin-disease-detection-
skin disease  and skin type detection, skin care recommendation
# 🧴 Skin Disease Detection and Skin Care Recommendation System using Deep Learning

## 📌 Project Description

The Skin Disease Detection and Skin Care Recommendation System is an AI-powered web application designed to detect skin diseases and identify skin types using deep learning techniques. The system uses Convolutional Neural Networks (CNN) to analyze uploaded skin images and predict the disease class and skin type accurately.

Based on the prediction, the system provides:

- Disease name with confidence score
- First-line treatment suggestions
- Do’s and Don’ts guidelines
- Skin care recommendations
- Morning and night skincare routine
- Recommended skincare products

The system helps users in early detection, awareness, and proper skin care guidance. It also demonstrates the real-world application of Artificial Intelligence in healthcare.

---

## 🎯 Objectives

- To develop an automated system for skin disease detection using deep learning.
- To classify skin type from facial images.
- To provide first-line treatment recommendations.
- To provide skincare routine and recommendations.
- To assist users in maintaining healthy skin.
- To create a user-friendly web-based interface.

---

## ❗ Problem Statement

Skin diseases are common worldwide, but early detection is difficult due to lack of awareness and access to dermatologists. Manual diagnosis is time-consuming and requires expert knowledge.

This project aims to develop an automated deep learning-based system that detects skin diseases and provides skincare recommendations using image analysis.

---

## 💡 Proposed Solution

The proposed system uses a Convolutional Neural Network (CNN) model trained on skin image datasets to classify skin diseases and skin types.

The system workflow:

1. User uploads a skin image
2. Image is preprocessed (resize, normalization)
3. CNN model analyzes the image
4. Disease and skin type are predicted
5. Results displayed on web page
6. Treatment, skincare routine, and recommendations are shown

---

## 🧠 Technologies Used

### Frontend
- HTML
- CSS
- Bootstrap
- JavaScript

### Backend
- Python
- Flask Framework

### AI / Machine Learning
- TensorFlow
- Keras
- NumPy
- OpenCV

### Development Tools
- Jupyter Notebook
- VS Code
- Git & GitHub

---

## 🧬 Dataset Description

The dataset contains skin images belonging to different disease classes and skin types.

### Disease Classes:
- Acne
- Eczema
- Psoriasis
- Vitiligo
- Melanoma
- Healthy

### Skin Type Classes:
- Oily
- Dry
- Normal

Images were preprocessed by:

- Resizing to 224 × 224 pixels
- Normalization
- Converting to array format

---

## 🤖 Model Used

Convolutional Neural Network (CNN)

### CNN Layers Used:
- Convolutional Layers
- ReLU Activation
- Max Pooling Layers
- Flatten Layer
- Dense Layers
- Softmax Output Layer

### Loss Function:
Categorical Crossentropy

### Optimizer:
Adam Optimizer

---

## 🔄 System Architecture

User → Web Interface → Flask Backend → CNN Model → Prediction → Result Page

---

## ⚙️ Project Workflow

1. User uploads image through web interface
2. Image preprocessing is performed
3. CNN model predicts disease and skin type
4. System displays prediction results
5. First-line treatment is shown
6. Do’s and Don’ts are displayed
7. Skincare routine is provided
8. Product recommendations are shown

---

## 📊 Features

- Skin Disease Detection
- Skin Type Classification
- First-line Treatment Suggestions
- Do’s and Don’ts Guidelines
- Morning and Night Routine
- Product Recommendations
- User-Friendly Interface

---

## 📁 Project Structure

finalyearproject/
│
├── app.py
├── classes.json
├── models/
│ ├── disease_model.h5
│ └── skintype_model.h5
│
├── static/
│ ├── uploads/
│ └── css/
│
├── templates/
│ ├── index.html
│ ├── disease.html
│ ├── disease_result.html
│ ├── skin_type.html
│ └── skin_type_result.html
│
└── README.md
