from zenml.steps import step

@step
def preprocessor_step(X_train, X_test, scaler):
    return scaler.transform(X_train), scaler.transform(X_test)
