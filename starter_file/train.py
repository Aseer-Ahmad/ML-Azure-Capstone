import argparse
import os
import numpy as np
import joblib
import pandas as pd

from azureml.core.run import Run
from azureml.core import Workspace
from azureml.core import Dataset

from sklearn import datasets, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

ws = Workspace(
    subscription_id="051560d6-9344-4907-a9c5-057add5cf030",
    resource_group="udacity_rg",
    workspace_name="udacity_ws"
)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="")
    parser.add_argument('--max_depth', type=int, default=4, help="")
    parser.add_argument('--min_samples_split', type=int, default=5, help="")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="")
    
    args = parser.parse_args()

    run = Run.get_context()

    run.log("n_estimators:",      np.int(args.n_estimators))
    run.log("max_depth:",         np.int(args.max_depth))
    run.log("min_samples_split:", np.int(args.min_samples_split))
    run.log("learning_rate:",     np.float(args.learning_rate))
    
    dataset_name = 'denver-cpi'
    ds = Dataset.get_by_name(ws, databset_name)
    df = ds.to_pandas_dataframe()
    y = df['cpi']
    x = df.drop(columns=['cpi'])
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=0)

    params = {
        "n_estimators": np.int(args.n_estimators),
        "max_depth": np.int(args.max_depth),
        "min_samples_split": np.int(args.min_samples_split),
        "learning_rate": np.float(args.learning_rate),
        "loss": "squared_error",
    }
    
    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(X_train, y_train)
    
    mse = mean_squared_error(y_test, reg.predict(X_test))    
    run.log("MSE", np.float(mse))

if __name__ == '__main__':
    main()