import argparse
import os
import yaml 
import pandas as pd 


def read_params(config_path):
    '''
    Read the params.yaml file and return a dictionary with various parameters along with its values
    '''

    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    
    return config 


def get_data_frame(config_path):
    '''
    Read the CSV file from the local directory and return a pandas dataframe
    '''
    config = read_params(config_path)

    data_path = config['data_source']['local_data_source']['process_data_source']
    sample_ = config['data_source']['local_data_source']['data_sample']

    df = pd.read_csv(data_path, sep=',')

    return df          #df.sample(n=sample_)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    

