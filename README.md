# Emotion Detection (FER2013 + CNN)

## Setup

1. Create & activate venv
2. pip install -r requirements.txt
3. put fer2013.csv in the data/ folder

## Train

python src/train.py

## Evaluate

python src/evaluate.py

## Run real-time demo

python src/inference.py

## Notes

- Model saved to models/best_model.h5
- Use GPU for faster training (Colab recommended)
