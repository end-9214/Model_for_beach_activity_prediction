from zenml.steps import step
import torch
from torch.utils.data import DataLoader, Dataset
import mlflow
from Beach_Activity_Prediction_model.models.model import *

class BeachActivityDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

@step
def trainer_step(processed_data, y_train):
    input_dim = processed_data.shape[1]
    hidden_dim = 64
    output_dim = len(set(y_train))

    train_dataset = BeachActivityDataset(processed_data, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = ActivityLevelClassifier(input_dim, hidden_dim, output_dim)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # MLflow Experiment
    mlflow.start_run()
    mlflow.log_param("input_dim", input_dim)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("output_dim", output_dim)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)

    # Training loop
    for epoch in range(50):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Log epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        mlflow.log_metric("epoch_loss", avg_loss, step=epoch)

    # Save the model
    mlflow.pytorch.log_model(model, artifact_path="models")

    mlflow.end_run()

    return model
