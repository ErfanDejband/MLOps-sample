import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- Configuration ---
DATA_PATH = 'data/data.csv'
MODEL_SAVE_PATH = 'model/simple_dnn_model.h5'
RANDOM_SEED = 42

# --- 1. Load Data ---
df = pd.read_csv(DATA_PATH)
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

# --- 2. Build and Train Simple DNN ---
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') # Output layer for binary classification
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

history = model.fit(
    X_train, y_train, 
    epochs=10, 
    batch_size=32, 
    verbose=0 # Run silently
)

# --- 3. Calculate Metric ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Model Training Complete.")
print(f"Test Accuracy: {accuracy:.4f}")

# --- 4. Save the Model ---
# Create the model directory if it doesn't exist
import os
os.makedirs('model', exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")