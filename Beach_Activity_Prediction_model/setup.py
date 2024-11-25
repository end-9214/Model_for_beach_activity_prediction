from setuptools import setup, find_packages

setup(
    name="Beach_Activity_Prediction_model",
    version="1.0.0",
    author="Karamveer Singh",
    author_email="karamveersingh2003111@gmail.com",
    description="A production-ready machine learning pipeline for predicting beach activity levels.",
    long_description=open("ReadMe.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/end-9214/Beach_Activity_Prediction_model",
    packages=find_packages(include=["Beach_Activity_Prediction_model", "Beach_Activity_Prediction_model.*"]),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "seaborn",
        "zenml[mlflow]>=0.41.0",
        "mlflow",
    ],
    entry_points={
        "console_scripts": [
            "run-training-pipeline=Beach_Activity_Prediction_model.pipelines.training_pipeline:main",
        ]
    },
)
