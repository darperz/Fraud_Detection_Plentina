# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:07:43 2022

This script tests run.py by sending POST request

@author: Darwin Perez
"""


import requests
i = 0
while i < 5:
    r = requests.post('http://127.0.0.1:5000/is-fraud', json = {"step":1,"type":"PAYMENT","amount":9839.64,"nameOrig":"C1231006815","oldbalanceOrig":170136.0,"newbalanceOrig":160296.36,"nameDest":"M1979787155","oldbalanceDest":0.0,"newbalanceDest":0.0})
    
    print(r.json())

    i += 1