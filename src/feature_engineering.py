import argparse
import pandas as pd
import os
import sys
import csv
import joblib


from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import set_config
import category_encoders as ce

from get_data_ import read_params, get_data_frame

# set_config(transform_output="pandas")

def feature_engine(config_path):
    config = read_params(config_path)

    df = get_data_frame(config_path)

    # Handling the Feature : online_order
    '''
    One-Hot Encode the Feature : online_order
    '''
    df = pd.get_dummies(
        df, columns=['online_order'], drop_first=True, dtype=float)

    # Handling the Feature : book_table
    '''
    One-Hot Encode the Feature : book_table
    '''
    df = pd.get_dummies(
        df, columns=['book_table'], drop_first=True, dtype=float)

    # Handling the Feature : location
    '''
    Encode the Feature : location -> Combine levels: To avoid redundant levels in a categorical variable and to deal with rare levels, we can simply combine the different levels. There are various methods of combining levels. Here are commonly used ones:

Using Business Logic: It is one of the most effective method of combining levels. It makes sense also to combine similar levels into similar groups based on domain or business experience. For example, we can combine levels of a variable “zip code” at state or district level.
    
    This will reduce the number of levels and improve the model performance also.
    '''
    pincode_csv_loc = config['category_encoding']['pincode_location']

    pincode_csv_df = pd.read_csv(pincode_csv_loc)

    pincode_csv_df.drop(columns=['Unnamed: 0'],inplace=True)

    pincode_dict = pincode_csv_df.set_index('location')['pincode'].to_dict()

    df['loc_pincode'] = df['location'].map(pincode_dict)

    
    # Handling the Feature : rest_type
    '''
    Multilabel Binarizer
    '''
    df['rest_type'] = df['rest_type'].apply(lister)

    

    # df.to_csv('datagrame.csv', index=False)


    mlb = MultiLabelBinarizer()
    # mlb.set_output(transform="pandas")

    a = mlb.fit_transform(df['rest_type'])
    # a.to_csv('a.csv')

    cols = []
    for idx, val in enumerate(mlb.classes_):
        val = f"type_{val}"
        cols.append(val)

    df[cols] = a

    rest_type_mlb_path = config['ml_flow_config']['rest_type_encoder']
    joblib.dump(mlb, rest_type_mlb_path) # save the model
    # # print(cols)

    # b = pd.DataFrame(a, columns=cols)

    # b.to_csv('b.csv', index=False)

    # df1 = pd.merge(df, b, left_index=True, right_on=b.index, sort=False)
    # # df1 = df.join(b)
    # # df1 = df.merge(b, left_index=True, right_index=True, on=)
    # # df1 = pd.concat([df, b], ignore_index=True, axis=1, sort=False)

     #------------------------------------------------------------------

#     # Handling the Feature : cuisines
#     '''
#     Binary Encoding

# Binary encoding is a combination of Hash encoding and one-hot encoding. In this encoding scheme, the categorical feature is first converted into numerical using an ordinal encoder. Then the numbers are transformed in the binary number. After that binary value is split into different columns.

# Binary encoding works really well when there are a high number of categories.
#     '''
    df_cuisine = pd.DataFrame(df['cuisines'])
    encoder_ce = ce.BinaryEncoder(cols=['cuisines'], return_df=True)

    df_cuisine = encoder_ce.fit_transform(df_cuisine)

    df = pd.concat([df_cuisine, df], axis="columns")

#     #   DROP THE UNWANTED COLUMNS
    cuisine_binary_encoder_path = config['ml_flow_config']['cuisine_encoder']
    joblib.dump(encoder_ce, cuisine_binary_encoder_path) # save the model
    
    df.drop(columns=['type', 'location', 'rest_type', 'cuisines'], inplace=True)

    train_eval_csv_path = config['data_source']['local_data_source']['processed_data_train_and_eval']
    df.to_csv(train_eval_csv_path, index=False)   

    print("\nFinished Feature Engineering\n")
            
def lister(val):
    '''
    this function takes the value and return it in a list
    '''
    if ',' in val:
        ls = val.split(', ')
        return ls
    else:
        return [val]





if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    feature_engine(parsed_args.config)
