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

---

## 4. CNN Architecture

Input: 64×64×3  

Conv2D(32) → MaxPool  
Conv2D(64) → MaxPool  
Conv2D(128) → MaxPool  
Flatten  
Dense(128) → Dropout(0.5)  
Dense(10) → Softmax  

---

## 📊 Training Epoch Results

Epoch 1/10  
accuracy: 0.4364 - loss: 1.5371 - val_accuracy: 0.6917 - val_loss: 0.8460  

Epoch 2/10  
accuracy: 0.7727 - loss: 0.6428 - val_accuracy: 0.8257 - val_loss: 0.4961  

Epoch 3/10  
accuracy: 0.8643 - loss: 0.3835 - val_accuracy: 0.7193 - val_loss: 0.8146  

Epoch 4/10  
accuracy: 0.9101 - loss: 0.2690 - val_accuracy: 0.7810 - val_loss: 0.6635  

Epoch 5/10  
accuracy: 0.9287 - loss: 0.2177 - val_accuracy: 0.8457 - val_loss: 0.4952  

Epoch 6/10  
accuracy: 0.9456 - loss: 0.1671 - val_accuracy: 0.9180 - val_loss: 0.2533  

Epoch 7/10  
accuracy: 0.9482 - loss: 0.1520 - val_accuracy: 0.8235 - val_loss: 0.5404  

Epoch 8/10  
accuracy: 0.9605 - loss: 0.1221 - val_accuracy: 0.6992 - val_loss: 0.8933  

Epoch 9/10  
accuracy: 0.9619 - loss: 0.1087 - val_accuracy: 0.9233 - val_loss: 0.3676  

Epoch 10/10  
accuracy: 0.9679 - loss: 0.0996 - val_accuracy: 0.8512 - val_loss: 0.6139  

---

# 📊 Evaluation Results

## 🔥 Training Performance
- Final Training Accuracy: 96.79%
- Final Training Loss: 0.0996

---

## 🧪 Validation Performance (Best Epoch)
- Best Validation Accuracy: 92.33% (Epoch 9)
- Best Validation Loss: 0.2533

---

## 🧾 Test Performance
- Test Accuracy: 85.12%
- Test Loss: 0.6139

---

## 📌 Classification Report (Test Set)

              precision    recall  f1-score   support

           0       0.89      1.00      0.94       400
           1       1.00      0.98      0.99       400
           2       0.74      0.73      0.73       400
           3       0.93      0.65      0.77       400
           4       0.87      0.71      0.79       400
           5       0.77      0.96      0.86       400
           6       0.90      1.00      0.95       400
           7       0.70      0.81      0.75       400
           8       0.89      1.00      0.94       400
           9       0.91      0.67      0.77       400

    accuracy                           0.85      4000
   macro avg       0.86      0.85      0.85      4000
weighted avg       0.86      0.85      0.85      4000

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

