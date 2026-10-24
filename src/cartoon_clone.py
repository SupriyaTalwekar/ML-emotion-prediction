import cv2
import numpy as np

def draw_face(frame, emotion="neutral"):
    """
    Draws a cartoon face on a black frame based on the given emotion.
    """
    h, w, _ = frame.shape
    center = (w//2, h//2)
    radius = 100

    # Head
    cv2.circle(frame, center, radius, (0, 255, 255), -1)  # yellow face

    # Eyes
    eye_y = center[1] - 30
    eye_x_offset = 40
    cv2.circle(frame, (center[0] - eye_x_offset, eye_y), 10, (0,0,0), -1)
    cv2.circle(frame, (center[0] + eye_x_offset, eye_y), 10, (0,0,0), -1)

    # Mouth position
    mouth_y = center[1] + 40

    # Draw mouth based on emotion
    if emotion == "happy":
        cv2.ellipse(frame, (center[0], mouth_y), (40, 20), 0, 0, 180, (0,0,0), 3)
    elif emotion == "sad":
        cv2.ellipse(frame, (center[0], mouth_y+20), (40, 20), 0, 180, 360, (0,0,0), 3)
    elif emotion == "angry":
        cv2.line(frame, (center[0]-40, mouth_y+10), (center[0]+40, mouth_y), (0,0,0), 3)
    elif emotion == "surprise":
        cv2.circle(frame, (center[0], mouth_y), 15, (0,0,0), 3)
    elif emotion == "fear":
        cv2.ellipse(frame, (center[0], mouth_y), (30, 10), 0, 0, 360, (0,0,0), 2)
    elif emotion == "disgust":
        cv2.line(frame, (center[0]-40, mouth_y), (center[0]+40, mouth_y+10), (0,0,0), 3)
    else:  # neutral
        cv2.line(frame, (center[0]-40, mouth_y), (center[0]+40, mouth_y), (0,0,0), 3)

    # Draw emotion label on top
    cv2.putText(frame, f'Emotion: {emotion}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return frame
