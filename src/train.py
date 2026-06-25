# Importing the required libraries
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv
load_dotenv()
import numpy as np
from torch.utils.data import DataLoader


if __package__ is None and __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from src.dataset import load_and_prepare_data, CMAPSSDataset
from src.model import LSTMRULRegressor

# Using DagsHub for data version control
DAGSHUB_USERNAME = os.environ.get("DAGSHUB_USERNAME")
DAGSHUB_PASSWORD = os.environ.get("DAGSHUB_PASSWORD") or os.environ.get("DAGSHUB_TOKEN")
REPO_NAME = os.environ.get("DAGSHUB_REPO_NAME", "nasa-rul-lstm")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

# Setup of MLflow for experiment tracking and logging
def setup_mlflow():
    if MLFLOW_TRACKING_URI:
        print(f"Using environment MLflow tracking URI: {MLFLOW_TRACKING_URI}")
        if DAGSHUB_USERNAME and DAGSHUB_PASSWORD:
            os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
            os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_PASSWORD
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    elif DAGSHUB_USERNAME and DAGSHUB_PASSWORD:
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_PASSWORD
        uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow"
        print(f"Using DagsHub MLflow tracking URI: {uri}")
        mlflow.set_tracking_uri(uri)
    else:
        local_uri = "sqlite:///mlflow.db"
        print(f"Using local MLflow SQLite backend: {local_uri}")
        mlflow.set_tracking_uri(local_uri)

    try:
        # Setting MLFLOW experiment
        mlflow.set_experiment("NASA_Turbofan_RUL_Prediction") 
    except Exception as e:
        print(f"MLflow setup failed: {e}\nFalling back to local SQLite backend.")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("NASA_Turbofan_RUL_Prediction")

setup_mlflow()

 

# Training loop and Hyperparameter Tunning 
def train_model():
    # Hyperparameters
    SEQ_LENGTH = 30 # Sequence length
    BATCH_SIZE = 64 # Batch size
    EPOCHS = 30 # Epochs 
    LR = 0.001   # Learning Rate
    HIDDEN_DIM = 64 # Hidden dimensions
    NUM_LAYERS = 2 # Layers
    
    # Ensuring existance of data directory
    if not os.path.exists('data/train_FD001.txt'):
        print("Error: Dataset files not found in 'data/' folder. Please download CMAPSS dataset.")
        return

    # Load and scale standard subsets (using FD001 as baseline example)
    X_train, y_train, X_test, y_test, _, _ = load_and_prepare_data(
        'data/train_FD001.txt', 'data/test_FD001.txt', 'data/RUL_FD001.txt', SEQ_LENGTH
    )
    # Train and Test loaders
    train_loader = DataLoader(CMAPSSDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(CMAPSSDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    # Device which will be used in PyTorch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Model class
    model = LSTMRULRegressor(input_dim=24, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # We used Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Start tracking session within MLflow
    with mlflow.start_run():
        # Logging parameters to mlflow ui
        mlflow.log_param("sequence_length", SEQ_LENGTH)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LR)
        mlflow.log_param("hidden_dimension", HIDDEN_DIM)
        mlflow.log_param("lstm_layers", NUM_LAYERS)
        
        print("Training engine lifecycle initialized...")

        # It will learn untill the last epoch 
        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss = 0.0 #initial loss
            for sequences, targets in train_loader:
                # Adding sequence and targets in device
                sequences, targets = sequences.to(device), targets.to(device)
                
                # Resetting gradients every time
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                # Backward propogation
                loss.backward()
                # Updatting W's
                optimizer.step()
                # Total Training loss
                train_loss += loss.item() * sequences.size(0)
        
            train_loss /= len(train_loader.dataset)
            
            # Evaluating and validating model
            model.eval()
            test_loss = 0.0
            # learing si off
            with torch.no_grad():
                for sequences, targets in test_loader:
                    sequences, targets = sequences.to(device), targets.to(device)
                    outputs = model(sequences)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item() * sequences.size(0)
            test_loss /= len(test_loader.dataset)
            
            rmse_val = np.sqrt(test_loss)  # Mean squared error on test loss
            
            # Logging metrics on mlflow ui
            mlflow.log_metric("train_mse", train_loss, step=epoch)
            mlflow.log_metric("test_mse", test_loss, step=epoch)
            mlflow.log_metric("test_rmse", rmse_val, step=epoch)
            
            print(f"Epoch {epoch:02d}/{EPOCHS} | Train MSE: {train_loss:.2f} | Test MSE: {test_loss:.2f} | Test RMSE: {rmse_val:.2f}")
            
        # Log the model to MLflow.
        # Use model registry only when a remote MLflow store is configured.
        if MLFLOW_TRACKING_URI or (DAGSHUB_USERNAME and DAGSHUB_PASSWORD):
            try:
                mlflow.pytorch.log_model(
                    model,
                    artifact_path="lstm_rul_model",
                    registered_model_name="LSTM_RUL_Turbofan",
                )
            except Exception as e:
                print(f"Remote model registry unavailable: {e}\nLogging model artifact locally instead.")
                mlflow.pytorch.log_model(model, artifact_path="lstm_rul_model")
        else:
            mlflow.pytorch.log_model(model, artifact_path="lstm_rul_model")

        # Save a copy locally to deploy using FastAPI
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/lstm_weights.pth")
        print("Model trained and safely saved to models/lstm_weights.pth.")

if __name__ == "__main__":
    train_model()