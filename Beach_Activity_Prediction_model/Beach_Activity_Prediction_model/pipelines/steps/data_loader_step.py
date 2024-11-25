from zenml.steps import step
from Beach_Activity_Prediction_model.dataset.data_preparation import load_and_preprocess_data


@step
def data_loader_step():
    return load_and_preprocess_data()
