import argparse 
import pandas as pd 
import os 
import sys 

from get_data_ import read_params, get_data_frame 


def feature_engine(config_path):
    config = read_params(config_path)

    df = get_data_frame(config_path)

    




















if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    feature_engine(parsed_args.config)