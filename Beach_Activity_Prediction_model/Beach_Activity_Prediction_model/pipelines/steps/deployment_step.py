from zenml.steps import step
import mlflow

@step
def deployment_step(model):
    mlflow.start_run()
    mlflow.pytorch.log_model(model, artifact_path="deployed_model")
    mlflow.end_run()
