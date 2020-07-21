# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 01:09:06 2020

@author: Rohit
"""

import requests

url = 'http://localhost:5000/predict'
r = requests.post(url,json={'Area':750, 'BHK':3, 'Bathroom':2, 'Parking':1, 'Per_Sqft':6667, 'Type_Builder_Floor':1})

print(r.json())