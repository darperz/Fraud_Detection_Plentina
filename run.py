# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:01:36 2022

This script runs the REST-API for Fraud-Detection with one endpoint
for POST method

@author: Darwin Perez
"""
from flask import Flask
app = Flask(__name__)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import Fraud_Detec_utils
import time
import pickle

glob_call_counter = 0
glob_archive_feature_df = pd.DataFrame()

# Setup route or endpoint at this dir
@app.route('/is-fraud',methods=['POST'])
def home():
    start_time = time.time()
    input_feature_dict = {}
    input_feature_df = pd.DataFrame()
    
    # Load the model
    model = pickle.load(open('rf_model.pkl', 'rb'))

    # Get Input data using JSON
    input_feature_dict = Fraud_Detec_utils.parse_json()
    
    ### User-input Preprocessing ###
    # Convert dict to dataframe for input to model
    # df needs atleast 2 index. Just drop the next index
    input_feature_df = pd.DataFrame(input_feature_dict, index=(0,1))
    input_feature_df = input_feature_df.drop(1)
    
    # Change features: type, nameOrig and nameDest to float
    input_feature_df['type'] = le.fit_transform(input_feature_df['type'])
    input_feature_df['nameOrig'] = le.fit_transform(input_feature_df['nameOrig'])
    input_feature_df['nameDest'] = le.fit_transform(input_feature_df['nameDest'])
    
    # Parse to model
    prediction = model.predict(input_feature_df)
    
    # Create input_feature_df with the prediction result/isFraud col
    # and dictionary
    if prediction[0] == 0:
        prediction = {"isFraud":False}
        input_feature_df['isFraud'] = 0
    else: 
        prediction = {"isFraud":True}
        input_feature_df['isFraud'] = 1
    
    print("POST count:",glob_call_counter)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return prediction
    

# Run the application
if __name__ == '__main__':
   app.run()
   
