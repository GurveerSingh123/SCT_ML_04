import cv2
import numpy as np
import tensorflow as tf
import joblib
from collections import deque

model=tf.keras.models.load_model('gesture_model.h5')
encoder=joblib.load('label_encoder.pkl')

IMG_SIZE=64
cap=cv2.VideoCapture(0)
buffer = deque(maxlen=10)
while True:
    ret,frame=cap.read()
    
    if not ret:
        break

    frame=cv2.flip(frame,1)
    roi = frame[100:300, 100:300]

    img=cv2.resize(roi,(IMG_SIZE,IMG_SIZE))
    img=img/255.0
    img=np.expand_dims(img,axis=0)

    pred = model.predict(img)
    confidence = np.max(pred)
    class_id = np.argmax(pred)


    if confidence >= 0.8:
        buffer.append(class_id)
        final_prediction = max(set(buffer), key=buffer.count)
        gesture = encoder.inverse_transform([final_prediction])[0]
    else:
        gesture = "Unknown"
    # Show text
    cv2.rectangle(frame, (100,100), (300,300), (255,0,0), 2)
    cv2.putText(frame, gesture, (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()