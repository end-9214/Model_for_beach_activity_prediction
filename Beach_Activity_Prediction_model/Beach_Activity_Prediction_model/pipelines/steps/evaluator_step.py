from zenml.steps import step
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

@step
def evaluator_step(model, X_test, y_test, label_encoder):
    model.eval()

    # Convert test data to torch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(y_test_tensor.numpy())

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_)
    cm = confusion_matrix(all_labels, all_preds)

    # Log metrics and artifacts to MLflow
    mlflow.start_run()
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_text(report, "classification_report.txt")

    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    mlflow.end_run()
