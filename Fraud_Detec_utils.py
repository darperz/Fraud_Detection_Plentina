# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 15:01:39 2022

This file contains functions used by several scripts in the
project

@author: Darwin Perez
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from flask import request
from sklearn.feature_selection import RFECV


"""
Function contains data processing to better understand
the contents of the data
ARG: (str): location of the data file 
RETURN: None
"""
def data_processing(df):
    print(type(df))
    print("Head")
    print(df.head())
    print()
    
    # Change string to numeric values
    df['type'] = le.fit_transform(df['type'])
    df['nameOrig'] = le.fit_transform(df['nameOrig'])
    df['nameDest'] = le.fit_transform(df['nameDest'])
    
    print("Head after LabelEncoder")
    print(df.head())
    print()
    
    # Checking the shape
    print("Shape")
    print(df.shape)
    print()
    
    # Checking the datatypes and null/non-null distribution
    print("Info/data types and null/non-null distribution")
    print(df.info())
    print()
    
    # Checking distribution of numerical values in the dataset
    print("Describe/ Distribution of numerical values")
    print(df.describe())
    print()
    
    # Checking the class distribution of the target variable
    print("Distribution of non-fraud and fraud")
    print(df['isFraud'].value_counts())
    print()
    
    # Checking the class distribution of the target variable in percentage
    print("% of non-fraud and fraud")
    print((df.groupby('isFraud')['isFraud'].count()/df['isFraud'].count()) *100)
    ((df.groupby('isFraud')['isFraud'].count()/df['isFraud'].count()) *100).plot.pie()
   
    
"""  
Function contains splitting of data to training and test
ARG: (str): location of the data file 
RETURN: 
    (pandas.core.frame.DataFrame) X_train 
    (pandas.core.series.Series) Y_train
    (pandas.core.frame.DataFrame) X_test
    (pandas.core.series.Series) Y_test
"""
def data_splitting(df):
    # Splitting the dataset into X and y
    Y = df['isFraud']
    X = df.drop(['isFraud'], axis=1)
    
    # Checking some rows of X and Y
    print()
    print("X head or other attributtes")
    print(X.head())
    print()
    print("Y head or is-fraud value")
    print(Y.head())
    print()
    
    # Splitting the dataset using train test split. 
    # Default test size is 25% of Y
    # Use random_state=100 to reproduce output
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=100)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    
    # Checking the spread of data post split
    print("Sum of Y or Total is fraud (value=1):",np.sum(Y))
    print("is-fraud in Y_train:", np.sum(Y_train))
    print("is-fraud in Y_test:",np.sum(Y_test))
    
    print("shapes:")
    print("X_train", X_train.shape)
    print("X_test", X_test.shape)
    print("Y_train", Y_train.shape)
    print("Y_test", Y_test.shape)
    
    return X_train, X_test, Y_train, Y_test


"""  
Function preprocess, run and check the accuracy of the model
ARG: (pandas.core.frame.DataFrame): archive_feature_df
RETURN: 
    (sklearn.ensemble._forest.RandomForestClassifier) rf 
Note: Default Roc Auc Score: 0.8770340809448801
"""
def model_run(archive_feature_df):
    # Perform data processing
    data_processing(archive_feature_df)
        
    X_train, X_test, Y_train, Y_test = data_splitting(archive_feature_df)
    
    # Grid Search hyperparameter tuned RandomForest model
    # n_jobs=(-1) is number of jobs to run in parallel, -1 means all
    # RFECV cv is default so it can auto b/n 5-fold and stratifiedKFold
    rf = RFECV(RandomForestClassifier(
        criterion = 'entropy',
        max_depth = 3,
        n_jobs=(-1)
        ))
    #rf = RandomForestClassifier(n_jobs=(-1))
    rf.fit(X_train, Y_train)
                  
    Y_pred = rf.predict(X_test)
    
    # Range is 0 to 1
    # All default RandomForestClassifier already gets 0.88 which is already good
    print("Roc Auc Score:", roc_auc_score(Y_test, Y_pred)) 
    # Range 0 to 1
    print("Score:",rf.score(X_test, Y_test)) 
    
    return rf
    
    
"""  
Function parses input data from html as a dictionary
ARG: NONE
RETURN: 
    (python dictionary) input_feature_dict 
"""
def parse_html():
    input_feature_dict = {}
    for key, val in request.form.items():
            input_feature_dict[key] = val
    return input_feature_dict
    

"""  
Function parses JSON input data from POST as a dictionary
ARG: NONE
RETURN: 
    (python dictionary) input_feature_dict 
"""
def parse_json():
    input_feature_dict = {
                'step':request.json['step'],
                'type':request.json['type'],
                'amount':request.json['amount'],
                'nameOrig':request.json['nameOrig'],
                'oldbalanceOrig':request.json['oldbalanceOrig'],
                'newbalanceOrig':request.json['newbalanceOrig'],
                'nameDest':request.json['nameDest'],
                'oldbalanceDest':request.json['oldbalanceDest'],
                'newbalanceDest':request.json['newbalanceDest']
            }
    return input_feature_dict