import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Paths
test_dir = "data/test"

# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48,48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

# Load trained model
model = load_model("models/best_model.h5")

# Predictions
y_true = test_generator.classes
y_pred = np.argmax(model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1), axis=1)

# Classification report
print(classification_report(y_true, y_pred, target_names=emotion_labels))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(range(7), emotion_labels, rotation=45)
plt.yticks(range(7), emotion_labels)
plt.tight_layout()
plt.show()
