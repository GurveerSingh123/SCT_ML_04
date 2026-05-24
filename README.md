# ✋ Hand Gesture Recognition using CNN

## 📌 Project Overview

This project builds a real-time hand gesture recognition system using a Convolutional Neural Network (CNN).  
It classifies hand gestures from images and enables live webcam-based gesture detection for human-computer interaction.

The model is trained on the LeapGestRecog dataset, which contains infrared hand images of multiple subjects performing 10 different gestures.

---

# 📂 Dataset Information

- Dataset: LeapGestRecog
- Total Gestures: 10 classes
- Subjects: 10 (00 to 09)
- Images: Infrared hand gesture images
- Format:
subject/gesture_name/frame_xxx.png

---

## 🧠 Gesture Classes
- palm
- L
- fist
- okay
- thumb
- down
- index
- call
- victory
- stop

---

# ⚙️ Project Workflow

## 1. Data Preprocessing
- Images resized to 64×64
- Normalized (0–1 scaling)
- Labels encoded using LabelEncoder
- One-hot encoding applied

## 2. Dataset Splitting
Subject-wise split:
- Training: 00–07
- Testing: 08–09

## 3. Data Augmentation
- Rotation: 25°
- Zoom: 0.2
- Width shift: 0.2
- Height shift: 0.2
- Horizontal flip

## 4. CNN Architecture

Input: 64×64×3

Conv2D(32) → MaxPool  
Conv2D(64) → MaxPool  
Conv2D(128) → MaxPool  
Flatten  
Dense(128) → Dropout(0.5)  
Dense(10) → Softmax  

## 5. Training
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 10
- Batch size: 32

---

# 📊 Evaluation Results

## Test Performance
- Accuracy: 94.55%
- Loss: 0.2355

## Observations
- Strong performance overall
- Some confusion in class 7 and 9
- Good generalization with subject-wise split

---

# 🎥 Real-Time System

Features:
- OpenCV webcam input
- ROI cropping
- Confidence thresholding
- Prediction smoothing (deque buffer)

---

# 🚀 How to Run

pip install numpy opencv-python tensorflow scikit-learn matplotlib seaborn joblib

python train_model.py
python webcam.py

---

# 💾 Outputs
- gesture_model.h5
- label_encoder.pkl

---

# 🔮 Future Improvements
- MediaPipe hand tracking
- MobileNetV2 transfer learning
- Web deployment (Streamlit/Flask)
- Better real-time smoothing

---

# 👨‍💻 Author
Hand Gesture Recognition Project using CNN
