import joblib
import json 
import os 
from src.get_data_ import read_params
import argparse
import numpy as np
import pandas as pd 
from pprint import pprint

from prediction_service.pred_transform import _transformer

import warnings
warnings.filterwarnings('ignore')

model_path = os.path.join("prediction_service","poduction_model.pkl") 



def form_response(df):
    try:
        config = get_args_config()
        
        prod_model_path = config['ml_flow_config']['production_model_path']


        
        data = pd.DataFrame(df, columns=df.keys(), index=[0]).copy()

        data_for_pred =  _transformer(data, config)

        prod_model = joblib.load(prod_model_path)

        pred = prod_model.predict(data_for_pred)
        print(pred)

        val = round(pred.tolist()[0], 2)
        
        return f"{val} / 5"

        

    except Exception as e:
        print(e)
        # error ={"error": "Something went wrong try again"}
        error = {"error": e}
        return f"Try Again"



def get_args_config():
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    config = read_params(config_path=parsed_args.config)

    return config


