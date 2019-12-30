from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
import pandas as pd
import os
import csv
from sklearn.externals import joblib
#data = pd.read_csv("heart.csv")
'''x = np.array(data.trestbps.values)
y = np.array(data.target.values)
z = np.array(data.age.values)
x = x.reshape((-1, 1))'''

model = joblib.load('model.pkl')
new_model = joblib.load('xgb_model.pkl')
a = np.array([57,0,1,130,236,0,0,174,0,1.1,1,1,2])
a = a.reshape(1,-1)
print('Tree prediction:',*model.predict(a))
print('XGBoot prediction:',*new_model.predict(a))
