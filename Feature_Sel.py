# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:51:54 2022

Result is:
[ True  True  True  True  True  True  True  True  True]
[1 1 1 1 1 1 1 1 1]
--- 19871.140011787415 seconds ---

All features are used

@author: Darwin Perez
"""

from sklearn.feature_selection import RFECV
import time
import Fraud_Detec_utils
import Fraud_Detec_cons
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

start_time = time.time()

# Read CSV
archive_feature_df = pd.read_csv(Fraud_Detec_cons.str_data_loc)

# Perform data processing including string to numeric
Fraud_Detec_utils.data_processing(archive_feature_df)
        
# Splitting the dataset into X and y
Y = archive_feature_df['isFraud']
X = archive_feature_df.drop(['isFraud'], axis=1)

# n_jobs=(-1) is number of jobs to run in parallel, -1 means all
# RFECV cv is default so it can auto b/n 5-fold and stratifiedKFold
rf = RFECV(RandomForestClassifier(n_jobs=(-1)),scoring='roc_auc')

rf = rf.fit(X, Y)

print(rf.support_)
print(rf.ranking_)


print("--- %s seconds ---" % (time.time() - start_time))