import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.model import build_model

# Paths to dataset
train_dir = "data/train"
test_dir  = "data/test"

# Image data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48,48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

# Build CNN model
model = build_model(input_shape=(48,48,1), num_classes=7)

# Create models folder
os.makedirs("models", exist_ok=True)

# Callbacks
checkpoint = ModelCheckpoint("models/best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
early = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    epochs=40,
    callbacks=[checkpoint, early]
)

# Save final model
model.save("models/final_model.h5")
print("Training complete. Model saved in models/final_model.h5")
