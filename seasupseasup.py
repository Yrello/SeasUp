import numpy as np
import pandas as pd
import scipy.stats as st
import sklearn
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def back_feature_elem(data_frame, dep_var, col_list):
	while len(col_list) > 0:
		model = sm.Logit(dep_var, data_frame[col_list])
		result = model.fit(disp=0)
		largest_pvalue = round(result.pvalues, 3).nlargest(1)
		if largest_pvalue[0] < (0.05):
			return result
		else:
			col_list = col_list.drop(largest_pvalue.index)


heart_df = pd.read_csv("framingham.csv")
heart_df.drop(['education'], axis=1, inplace=True)
heart_df.head()

heart_df.rename(columns={'male': 'Sex_male'}, inplace=True)

heart_df.head()

count = 0
for i in heart_df.isnull().sum(axis=1):
	if i > 0:
		count = count + 1
print('Total number of rows with missing values is ', count)
print('since it is only', round((count / len(heart_df.index)) * 100),
	  'percent of the entire dataset the rows with missing values are excluded.')

heart_df.dropna(axis=0, inplace=True)

heart_df.describe()

from statsmodels.tools import add_constant as add_constant

heart_df_constant = add_constant(heart_df)
heart_df_constant.head()


new_features = heart_df[['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose', 'TenYearCHD']]
x = new_features.iloc[:, :-1]
y = new_features.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=5)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

print(sklearn.metrics.accuracy_score(y_test, y_pred))
