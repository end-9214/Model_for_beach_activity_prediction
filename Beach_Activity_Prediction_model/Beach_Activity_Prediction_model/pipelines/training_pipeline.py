from zenml.pipelines import pipeline
from Beach_Activity_Prediction_model.pipelines.steps import (data_loader_step, preprocessor_step, trainer_step, evaluator_step, deployment_step)


@pipeline
def training_pipeline():
    X_train, X_test, y_train, y_test, scaler, label_encoder = data_loader_step()
    processed_data = preprocessor_step(X_train, X_test, scaler)
    model = trainer_step(processed_data, y_train)
    evaluator_step(model, X_test, y_test, label_encoder)
    deployment_step(model)
