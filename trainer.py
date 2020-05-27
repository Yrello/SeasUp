from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
import pandas as pd
import os
import csv

import xgboost
#from sklearn import cross_validation
from sklearn.metrics import accuracy_score

data = pd.read_csv("heart.csv")
'''x = np.array(data.trestbps.values)
y = np.array(data.target.values)
z = np.array(data.age.values)
x = x.reshape((-1, 1))'''


model = tree.DecisionTreeClassifier(criterion = 'gini')
x = data.values[:, 0:8]# 1,2,4,5,6,8
# age,sex,blood pressure,cholestoral,blood sugar,max heart rate
y = data.values[:,13]
trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.3)
model.fit(trainX, trainY)
new_model = xgboost.XGBClassifier()
new_model.fit(trainX, trainY)
#model.fit(x, y)
#model = LogisticRegression().fit(x, y)

r_sq = model.score(testX, testY)
print('coefficient of determination:', r_sq)
print('new coef of determination:', new_model.score(testX, testY))
data.plot(x='age',y='target',style='k--')
