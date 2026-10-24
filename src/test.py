import cv2
import numpy as np
from tensorflow.keras.models import load_model
from .cartoon_clone import draw_face

# Load trained model
model = load_model("models/final_model.h5")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Webcam setup
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

prev_emotion = "neutral"  # To smooth reactions

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    frame_small = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotion = prev_emotion

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        if roi_gray.size == 0:
            continue

        roi_gray = cv2.resize(roi_gray, (48,48))
        roi = roi_gray.astype('float')/255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        # Fast prediction
        prediction = model.predict_on_batch(roi)[0]
        new_emotion = emotion_labels[prediction.argmax()]

        # Update only if changed
        if new_emotion != prev_emotion:
            prev_emotion = new_emotion

        emotion = prev_emotion
        break  # Only consider first detected face

    # Draw cartoon clone
    cartoon_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cartoon_frame = draw_face(cartoon_frame, emotion)

    # Display webcam and clone
    cv2.imshow("Webcam", frame)
    cv2.imshow("AI Cartoon Clone", cartoon_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
