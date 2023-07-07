
import os
import pandas
import warnings
warnings.filterwarnings('ignore')
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime as dt

from get_data_ import read_params
import argparse
import joblib
import json

import mlflow 
from urllib.parse import urlparse

from model_training_and_hyperParameter_tuning import hyper_parameter_tuning
# from src.compare_model_plotter import CompareModels


dt_now = dt.now()
experi_time = dt_now.strftime("%m/%d/%Y")
run_time = dt_now.strftime("%m/%d/%Y, %H:%M:%S")
#-------------------PREDICTION METRICS---------------------------

def predict_on_test_data(model,X_test):
    y_pred = model.predict(X_test)
    return y_pred

def get_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': round(mae, 3), 'MSE': round(mse, 3), 'RMSE': round(rmse, 3), 'R2': round(r2, 3)}



def create_compare_plot(y_true, y_pred, compare_plot_path):
    from compare_model_plotter import CompareModels
    plot = CompareModels()
    plot.add(model_name='Random Forest Regression', y_test=y_true, y_pred=y_pred)
    plot.show(compare_plot_path)


#-----------------------------------------------------------------------

def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_and_eval_data_path = config["data_source"]["local_data_source"]['processed_data_train_and_eval']
    random_state = config["base"]["random_seed"]
    test_size = config["base"]["test_size"]
    # model_dir = config["model_dir"]

    compare_plot_path = config['metrics_path']['compare_plot_path']
    random_seed = config['base']['random_seed']

    

    df = pd.read_csv(train_and_eval_data_path, sep=',')
    X = df.drop(['rate'], axis=1)
    y = df['rate']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state= random_seed)

#----------------ML FLOW-------------------
    
    ml_flow_config = config["ml_flow_config"]
    remote_server_uri = ml_flow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)

    mlflow.set_experiment(f"{ml_flow_config['experiment_name']} {experi_time}")

    with mlflow.start_run(run_name=f"{ml_flow_config['run_name']} {run_time}") as mlops_run:

        # best_params = hyper_parameter_tuning(X_train, y_train, random_seed)  # Hyper Parameter Tuning
        best_params = {'n_estimators': 101, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 30, 'bootstrap': False}


        n_estimators = best_params['n_estimators']
        min_samples_split = best_params['min_samples_split']
        min_samples_leaf = best_params['min_samples_leaf']
        max_features = best_params['max_features']
        max_depth = best_params['max_depth']
        bootstrap = best_params['bootstrap']
        
        model_tuned = RandomForestRegressor(n_estimators = n_estimators, min_samples_split = min_samples_split,
                                            min_samples_leaf= min_samples_leaf, max_features = max_features,
                                            max_depth= max_depth, bootstrap=bootstrap) 
        model_tuned.fit(X_train, y_train)

        y_pred = predict_on_test_data(model_tuned, X_test)

        metrics = get_metrics(y_test, y_pred)

        create_compare_plot(y_test, y_pred, compare_plot_path)


# {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2), 'entropy': round(entropy, 2)}
    # -------------------------------------------------
        # Log Parameters and Metrics
        for param in best_params:
            mlflow.log_param(param, best_params[param])
        
        for key, val in metrics.items():
            mlflow.log_metric(key, val)
        
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(model_tuned, "RF Regressor", registered_model_name=ml_flow_config["registered_model_name"])

        else:
            mlflow.sklearn.load_model(model_tuned, "Regressor")

        if not config['metrics_path']['compare_plot_path'] == None:
            mlflow.log_artifact(config['metrics_path']['compare_plot_path'], 'Metrics_compare_Plot')

        print("\nFinished Train and Eval and Logged ML Flow Registry\n")
            

    # -------------------------------------------------


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(parsed_args.config)







