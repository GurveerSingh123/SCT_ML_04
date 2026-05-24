import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.utils import to_categorical

dataset_path = "leapGestRecog"
IMG_SIZE=64
train_subjects = ['00','01','02','03','04','05','06','07']
test_subjects  = ['08','09']

X_train, y_train = [], []
X_test, y_test = [], []

for subject in os.listdir(dataset_path):
    subject_path=os.path.join(dataset_path,subject)
    if not os.path.isdir(subject_path):
        continue

    if subject in train_subjects:
        target_X = X_train
        target_y = y_train
    elif subject in test_subjects:
        target_X = X_test
        target_y = y_test
    else:
        continue

    for gesture in os.listdir(subject_path):
        if '_' not in gesture:
            continue
        gesture_path=os.path.join(subject_path,gesture)

        label = gesture.split('_',1)[1]
        for image_name in os.listdir(gesture_path):
            image_path=os.path.join(gesture_path,image_name)
            image=cv2.imread(image_path)
            image=cv2.resize(image,(IMG_SIZE,IMG_SIZE))
            target_X.append(image)
            target_y.append(label)



encoder = LabelEncoder()

all_labels = np.array(y_train + y_test)

encoder.fit(all_labels)

y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


X_train = np.array(X_train) / 255.0
X_test = np.array(X_test) / 255.0

datagen = ImageDataGenerator(
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)


model=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(len(encoder.classes_), activation='softmax')
])

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=10
)

test_loss, test_acc = model.evaluate(X_test, y_test)

print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)

y_pred = model.predict(X_test)

y_pred_classes=np.argmax(y_pred,axis=1)
y_true=np.argmax(y_test,axis=1)

print(classification_report(y_true, y_pred_classes))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

model.save("gesture_model.h5")
joblib.dump(encoder,"label_encoder.pkl")