import pandas as pd
import numpy as np

from graphviz import Source
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import tree



Dataset = pd.read_csv("Iris.csv")
Dataset = Dataset.dropna()
print(Dataset.head())

print(Dataset.shape)

Dataset = Dataset.replace(to_replace ="Iris-setosa", value ="0")
Dataset = Dataset.replace(to_replace ="Iris-versicolor", value ="1")
Dataset = Dataset.replace(to_replace ="Iris-virginica", value ="2")

X = np.array(Dataset[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
Y = np.array(Dataset["Species"])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 100)

clf_gini = DecisionTreeClassifier(criterion = "gini", max_depth = 5, min_samples_leaf = 3, random_state = 100)
clf_gini.fit(X_train, Y_train)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth = 5, min_samples_leaf = 3, random_state = 100)
clf_entropy.fit(X_train, Y_train)

y_pred_gini = clf_gini.predict(X_test)
print ("Accuracy : ", accuracy_score(Y_test,y_pred_gini)*100)
print ("Report : ",  classification_report(Y_test, y_pred_gini))

y_pred_entropy = clf_entropy.predict(X_test)
print ("Accuracy : ", accuracy_score(Y_test,y_pred_entropy)*100)
print ("Report : ",  classification_report(Y_test, y_pred_entropy))



Source( tree.export_graphviz(clf_gini, out_file=None, feature_names=X.columns))
tree.plot_tree(clf_entropy)

