import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- NEW: Import MLflow ---
import mlflow
import mlflow.keras 

# --- Configuration & Hyperparameters ---
DATA_PATH = 'data/data.csv'
MODEL_NAME = "SimpleDNN_Classifier" # NEW: Name for the Model Registry
RANDOM_SEED = 42

# Define hyperparameters we want to track
# We can change these later to test different runs
HYPERPARAMS = {
    "epochs": 10,
    "batch_size": 32,
    "layer1_units": 64,
    "optimizer": "adam"
}

# --- Start MLflow Run ---
# Every piece of code inside this 'with' block will be logged together
with mlflow.start_run() as run:
    # Get the unique Run ID for future reference
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")

    # --- 1. Data Loading and Splitting (Same as before) ---
    df = pd.read_csv(DATA_PATH)
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # --- 2. Log Hyperparameters (NEW) ---
    mlflow.log_params(HYPERPARAMS)

    # --- 3. Build and Train Simple DNN (Using logged params) ---
    model = Sequential([
        Dense(HYPERPARAMS["layer1_units"], activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=HYPERPARAMS["optimizer"], 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    model.fit(
        X_train, y_train, 
        epochs=HYPERPARAMS["epochs"], 
        batch_size=HYPERPARAMS["batch_size"], 
        verbose=0
    )

    # --- 4. Calculate Metric ---
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")

    # --- 5. Log Metrics (NEW) ---
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_loss", loss)

    # --- 6. Log the Model to MLflow (NEW: Keras specific) ---
    # This saves the model file and necessary metadata for deployment
    mlflow.keras.log_model(model, "model_artifact", registered_model_name=MODEL_NAME)
    
    print(f"Model logged and registered under the name: {MODEL_NAME}")