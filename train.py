import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import os
import matplotlib.pyplot as plt
import json

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
    Dense(16, activation='relu'),
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
print("Model Training Complete.")
print(f"Test Accuracy: {accuracy:.4f}")

# --- 4. Save the Model ---
# Create the model directory if it doesn't exist

os.makedirs('model', exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# --- 5. save loss and epochs as json for visualization in DVC ---
history_data = []
for i in range(len(history.history['loss'])):
    history_data.append({
        "epoch": i + 1,  # Start epoch at 1
        "loss": history.history['loss'][i],
        "accuracy": history.history['accuracy'][i],
        # Add validation metrics here if available, e.g.,
        # "val_loss": history.history.get('val_loss', [None])[i], 
    })

with open('model/simulated_history.json', 'w') as f:
    json.dump(history_data, f, indent=4)

# --- save metrics and plot for visualization in DVC ---
# create metric for train validation/test loss and accuracy
metrics = {
    "train_accuracy": history.history['accuracy'][-1],
    "train_loss": history.history['loss'][-1],
    "test_accuracy": accuracy,
    "test_loss": loss
}

with open('model/metrics.json', 'w') as f:
    json.dump(metrics, f)
