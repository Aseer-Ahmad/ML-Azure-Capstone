# Machine Learning with Azure - Capstone Project

## Project Overview

This is the capstone project for the Udacity Machine Learning Engineer with Azure Nanodegree. The project demonstrates the process of training, evaluating, and deploying machine learning models using Azure Machine Learning services. Two primary methods are explored for model training and optimization: Azure Automated ML (AutoML) and Azure HyperDrive for hyperparameter tuning. The best model is then deployed as a web service.

## Dataset

This project utilizes the **Denver Consumer Price Index (CPI)** dataset, sourced from Kaggle. The dataset contains time-series data about the Consumer Price Index in Denver.

The primary task is a regression problem: to predict the `cpi` value based on the other features in the dataset. The data is accessed by downloading it from KaggleHub and registering it as a Tabular Dataset in the Azure ML Workspace for use in experiments.

## AutoML

Azure's Automated ML (AutoML) was used to automatically discover the best machine learning model and its corresponding hyperparameters for the CPI prediction task. AutoML iterates through a variety of algorithms and feature engineering steps to find the optimal pipeline without extensive manual intervention. The goal was to find a regression model that minimizes the prediction error.

The AutoML run was configured with the following settings:
- **Experiment Timeout:** Set to 30 minutes to control costs.
- **Primary Metric:** `normalized_root_mean_squared_error` was chosen to evaluate model performance. The goal is to minimize this metric.
- **Task:** Regression.

The best model found by AutoML was a `VotingEnsemble` which combines the predictions of multiple models to improve overall accuracy and robustness.

## Hyperdrive

HyperDrive was used to perform a more focused hyperparameter tuning process on a specific algorithm: the `GradientBoostingRegressor` from scikit-learn. This approach allows for fine-tuning a chosen model to achieve optimal performance.

The HyperDrive experiment was configured as follows:
- **Model:** `GradientBoostingRegressor`
- **Parameter Sampler:** `RandomParameterSampling` was used to efficiently search the hyperparameter space.
- **Hyperparameter Space:**
    - `n_estimators`: `choice(25, 50, 75, 100)`
    - `max_depth`: `choice(2, 3, 4)`
    - `min_samples_split`: `choice(3, 4, 5)`
    - `learning_rate`: `choice(0.1, 0.01)`
- **Termination Policy:** A `BanditPolicy` was configured with an `evaluation_interval` of 2 and a `slack_factor` of 0.1. This policy terminates runs that are not performing as well as the top-performing runs, saving compute resources.
- **Primary Metric:** `MSE` (Mean Squared Error), with a goal to `MINIMIZE`.
- **Compute Target:** A pre-configured `AmlCompute` cluster.

The best run from the HyperDrive experiment achieved an MSE of **2.248** with the following parameters:
- **n_estimators:** 100
- **max_depth:** 4
- **min_samples_split:** 4
- **learning_rate:** 0.1

## AutoML vs Hyperdrive

- **AutoML** provides a broad, automated search across different model types and preprocessing pipelines. It's excellent for quickly establishing a strong baseline model with minimal configuration. The best model it produced was a `VotingEnsemble`.
- **HyperDrive** offers a targeted approach to optimize a specific, pre-selected model (`GradientBoostingRegressor` in this case). It gives the user more control over the search space and algorithm.

For this particular problem, the model produced by the HyperDrive run was selected for deployment due to its performance and interpretability.

## Deployment

The best model from the HyperDrive run was registered in the Azure ML Model Registry. From the registry, the model was deployed as a web service.

The deployment process involved:
1.  **Registering the Model:** The trained model from the best HyperDrive run was saved and registered in the workspace.
2.  **Creating an Entry Script:** A `score.py` script was created to load the model and define how to process incoming requests and make predictions.
3.  **Defining an Environment:** A conda environment (`sk_dep.yaml`) was specified, listing all necessary dependencies like scikit-learn and azureml-defaults.
4.  **Deploying to Web Service:** The model was deployed to Azure Container Instances (ACI) as a REST endpoint. This endpoint can be queried with new data to get real-time CPI predictions.

To query the endpoint, a POST request with a JSON payload containing the feature data is sent to the service's scoring URI.

## Screencast

A screencast demonstrating the entire project is available. The video walks through:
- The setup and execution of the AutoML and HyperDrive experiments in the Azure ML Studio.
- The registration of the best-performing model.
- The deployment of the model as a web service.
- A live demonstration of sending a request to the deployed endpoint and receiving a prediction.

## Future Improvement

- **Deploy to AKS:** For a production-ready solution, the model could be deployed to Azure Kubernetes Service (AKS) instead of ACI to handle higher traffic loads and provide better scalability and security.
- **MLOps Pipeline:** Implement a full MLOps pipeline using Azure Pipelines or GitHub Actions to automate the retraining, evaluation, and redeployment of the model whenever new data becomes available.
- **Data Drift Monitoring:** Set up data drift monitoring on the deployed model to detect if the incoming data distribution changes over time, which would trigger a retraining pipeline.
- **Advanced Hyperparameter Tuning:** Explore more advanced sampling methods like Bayesian Sampling in HyperDrive for a potentially more efficient search of the hyperparameter space.