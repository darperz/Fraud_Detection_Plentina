# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:07:43 2022

This script is for hyperparameter tuning using Grid Search.
It fits the model on each and every combination of hyperparameters
possible and records the model performance. 

@author: Darwin Perez
"""

import pandas as pd
import Fraud_Detec_cons
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn import ensemble, model_selection
import time
import Fraud_Detec_utils

start_time = time.time()

# Read CSV
archive_feature_df = pd.read_csv(Fraud_Detec_cons.str_data_loc)
# Perform data processing including string to numeric
Fraud_Detec_utils.data_processing(archive_feature_df)

# Splitting the dataset into X and y
Y = archive_feature_df['isFraud']
X = archive_feature_df.drop(['isFraud'], axis=1)

# Use all cores
classifier = ensemble.RandomForestClassifier(n_jobs=-1)
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [1, 3, 5],
    "criterion": ["gini", "entropy"]
}

model = model_selection.GridSearchCV(
    estimator = classifier, 
    param_grid = param_grid,
    # same number of samples
    scoring = "roc_auc",
    # Messages
    verbose = 10,
    # 1 core only because rf is all core
    n_jobs = 1
    #cv is default so it can auto b/n 5-fold and stratifiedKFold
    )

model.fit(X,Y)

print(model.best_score_)
print(model.best_estimator_.get_params())
print("--- %s seconds ---" % (time.time() - start_time))