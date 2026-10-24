import cv2
import numpy as np
from tensorflow.keras.models import load_model

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Load trained model
model = load_model("models/best_model.h5")

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        try:
            roi = cv2.resize(roi, (48,48))
        except:
            continue
        roi = roi.astype('float32')/255.0
        roi = roi.reshape(1,48,48,1)
        preds = model.predict(roi)[0]
        label = emotion_labels[np.argmax(preds)]
        cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
