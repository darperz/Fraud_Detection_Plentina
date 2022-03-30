# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:12:06 2022

This script creates a pickle of the model with the final features and parameters

@author: Darwin D. Perez
"""

import time
import Fraud_Detec_utils
import Fraud_Detec_cons
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

start_time = time.time()

# Read CSV
archive_feature_df = pd.read_csv(Fraud_Detec_cons.str_data_loc)

# Perform data processing
Fraud_Detec_utils.data_processing(archive_feature_df)

# Splitting the dataset into X and y
Y = archive_feature_df['isFraud']
X = archive_feature_df.drop(['isFraud'], axis=1)

# Grid Search hyperparameter tuned RandomForest model
# n_jobs=(-1) is number of jobs to run in parallel, -1 means all
rf = RandomForestClassifier(
    bootstrap = True,
    ccp_alpha = 0.0,
    class_weight = None, 
    criterion = 'gini',
    max_depth = 3,
    max_features = 'auto',
    max_leaf_nodes = None,
    max_samples = None,
    min_impurity_decrease = 0.0,
    min_impurity_split = None,
    min_samples_leaf = 1,
    min_samples_split = 2,
    min_weight_fraction_leaf = 0.0,
    n_estimators = 100,
    n_jobs = -1,
    oob_score = False,
    random_state = None,
    verbose = 0, 
    warm_start = False
    )

rf = rf.fit(X, Y)
              
# Dump model to pkl file
pickle.dump(rf, open('rf_model.pkl','wb'))

print("--- %s seconds ---" % (time.time() - start_time))