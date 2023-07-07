import joblib
import json 
import os 
import sys
  


from src import get_data_ as gt
import argparse
import numpy as np
import pandas as pd 
from pprint import pprint





def _transformer(df, config):

    # config = gt.read_params(config_path)

    rest_encoder_path = config['ml_flow_config']['rest_type_encoder']
    cui_encoder_path = config['ml_flow_config']['cuisine_encoder']
    rest_encoder = joblib.load(rest_encoder_path)
    cuisine_encoder = joblib.load(cui_encoder_path)
    

    df['rest_type'] = df['rest_type'].apply(lister)
    df['cuisines'] = df['cuisines'].apply(spacer)

    b = rest_encoder.transform(df['rest_type'])

    cols = []
    for idx, val in enumerate(rest_encoder.classes_):
        val = f"type_{val}"
        cols.append(val)

    df[cols] = b

    d_cui = pd.DataFrame(df['cuisines'])

    d_cui = cuisine_encoder.transform(d_cui)

    df1 = pd.concat([d_cui, df], axis="columns")

    df1.drop(['rest_type', 'cuisines'], axis=1, inplace=True)

    df1 = df1.loc[:,['cuisines_0', 'cuisines_1', 'cuisines_2', 'cuisines_3', 'cuisines_4',
        'cuisines_5', 'cuisines_6', 'cuisines_7', 'cuisines_8', 'cuisines_9',
        'cuisines_10', 'cuisines_11', 'cost_per_person', 'online_order_Yes',
        'book_table_Yes', 'loc_pincode', 'type_Bakery', 'type_Bar',
        'type_Beverage Shop', 'type_Bhojanalya', 'type_Cafe',
        'type_Casual Dining', 'type_Club', 'type_Confectionery',
        'type_Delivery', 'type_Dessert Parlor', 'type_Dhaba',
        'type_Fine Dining', 'type_Food Court', 'type_Food Truck',
        'type_Irani Cafee', 'type_Kiosk', 'type_Lounge', 'type_Meat Shop',
        'type_Mess', 'type_Microbrewery', 'type_Pop Up', 'type_Pub',
        'type_Quick Bites', 'type_Sweet Shop', 'type_Takeaway']]
    
    # print(df1, end="\n\n")
    print(df1.columns)
    return df1






   




def lister(val):
    '''
    this function takes the value and return it in a list
    '''
    if ',' in val:
        ls = val.split(', ')
        return ls
    else:
        return [val]
    


def spacer(val):
    
    if ',' in val:
        
        n_val =str(val).replace(',', ', ', 20)
        return n_val
    else:
        return val
    



# if __name__ == "__main__":

#     d = {'online_order_Yes': '1', 'book_table_Yes': '0', 'cost_per_person': '500', 'rest_type': 'Casual Dining, Cafe', 'loc_pincode': '380068', 'cuisines': 'African,Andhra,Asian'}
#     data = pd.DataFrame(d, columns=d.keys(), index=[0]).copy()

#     args = argparse.ArgumentParser()
#     args.add_argument("--config", default="params.yaml")
#     parsed_args = args.parse_args()
#     pred_transformer(data, parsed_args.config)
