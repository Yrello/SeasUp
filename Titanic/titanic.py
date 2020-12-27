import pandas as pd
import numpy as np
import random as rnd
from scipy.stats import norm
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

data = train.copy()
data.info()

print(data.head(10))
data.isna().sum()
data.describe()
data_null = data.isna().sum()
data_null.plot.bar()

data_num = data.select_dtypes('number')
plt.figure(figsize=(20, 10))
sns.boxplot(data=data_num)

for col in ['Age', 'Fare']:
    plt.figure(figsize=(5, 5))
    sns.distplot(data_num[col], bins=50, kde=True)

sns.pairplot(data_num)

corr = data_num.corr()
sns.heatmap(corr, annot=True)

upper_boundary = data['Age'].mean() + 3 * data['Age'].std()
lower_boundary = data['Age'].mean() - 3 * data['Age'].std()
print('For Age: the upper boundary is ' + str(upper_boundary))
print('For Age: the lower boundary is ' + str(lower_boundary))

print('-------------------------------------')
for col in ['Age', 'Fare']:
    IQR = data[col].quantile(0.75) - data[col].quantile(0.25)
    lower_bridge = data[col].quantile(0.25) - (IQR * 1.5)
    upper_bridge = data[col].quantile(0.75) + (IQR * 1.5)
    print('for ', str(col), ' the lower bridge is ' + str(lower_bridge))
    print('for ', str(col), ' the upper bridge is ' + str(upper_bridge))

print('-------------------------------------')

for col in ['Age', 'Fare']:
    IQR = data[col].quantile(0.75) - data[col].quantile(0.25)
    lower_bridge = data[col].quantile(0.25) - (IQR * 3)
    upper_bridge = data[col].quantile(0.75) + (IQR * 3)
    print('for ', str(col), ' the lower bridge is ' + str(lower_bridge))
    print('for ', str(col), ' the upper bridge is ' + str(upper_bridge))

data.Age[data.Age >= 73] = 73
data.Fare[data.Fare >= 100] = 100
data.describe()

plt.figure(figsize=(20, 10))
sns.boxplot(data=data.select_dtypes('number'))

data['Cabin'].value_counts()

data['Embarked'].value_counts()

s_imputer = SimpleImputer(strategy='most_frequent')
I_imputer = IterativeImputer()


def missing_values(data):
    data['Cabin'].fillna('none', inplace=True)
    data['Age'] = I_imputer.fit_transform(data[['Age']])
    data[[col for col in data.columns if col not in ['Age', 'Cabin']]] = s_imputer.fit_transform(
        data[[col for col in data.columns if col not in ['Age', 'Cabin']]])
    return


missing_values(data)
data.info()

passenger_Id = data['PassengerId']
y_train = data['Survived']

data.describe(include=['O'])


def feature_extraction(data):
    data.drop(['Ticket', 'Cabin'], axis=1, inplace=True)
    name = pd.Series(data['Name']).str.split(',', expand=True)
    name_ = pd.Series(name[1]).str.split('.', expand=True)
    data.Name = name_[0]
    return


feature_extraction(data)

data[['Name', 'Survived']].groupby(['Name'], as_index=False).mean().sort_values(by='Survived', ascending=False)

data.drop(['Survived', 'PassengerId'], axis=1, inplace=True)
data.Name.value_counts()


def replace_rare(data):
    data['Name'].replace([' Lady', ' the Countess', ' Capt', ' Col',
                          ' Don', ' Major', ' Sir', ' Jonkheer', ' Dona'], ' Rare', inplace=True)

    data['Name'].replace(' Mlle', ' Miss', inplace=True)
    data['Name'].replace(' Ms', ' Miss', inplace=True)
    data['Name'].replace(' Mme', ' Mrs', inplace=True)
    print(data.Name.value_counts())
    return


replace_rare(data)

ohe = OneHotEncoder(handle_unknown='ignore')
col_transformer = make_column_transformer((OrdinalEncoder(categories=[['male', 'female'], ['S', 'C', 'Q']]),
                                           ['Sex', 'Embarked']), remainder='passthrough')


def categorical_encoding(data_):
    global data

    cat_name = ohe.fit_transform(data_['Name'].to_numpy().reshape(-1, 1)).toarray()
    ohe_df = pd.DataFrame(cat_name, columns=ohe.get_feature_names())
    data_ = pd.concat([data_, ohe_df], axis=1).drop(['Name'], axis=1)

    data__ = col_transformer.fit_transform(data_)
    data = pd.DataFrame(data__,
                        columns=['Sex', 'Embarked'] + [col for col in data_.columns if col not in ['Sex', 'Embarked']])

    return data


categorical_encoding(data)

cols = ['Age', 'Fare']
for col in cols:
    fig = plt.figure()

    sns.distplot(data[col], fit=norm);
    fig = plt.figure()
    stats.probplot(data[col], plot=plt)


def transform_data(data):
    data['Fare'], parameters = stats.boxcox(data['Fare'].replace(0, 1))
    data['Fare'], parameters = stats.boxcox(data['Fare'].replace(0, 1))
    return


transform_data(data)
for col in cols:
    fig = plt.figure()

    sns.distplot(data[col], fit=norm);
    fig = plt.figure()
    stats.probplot(data[col], plot=plt)

std_scaler = StandardScaler()
data.Age = std_scaler.fit_transform(data.Age.to_numpy().reshape(-1, 1))


def processing_pipeline(test):
    def missing_values_(test):
        test['PassengerId'], test['Survived'] = passengerid, y_train
        test['Cabin'].fillna('none', inplace=True)
        test['Age'] = I_imputer.transform(test[['Age']])
        test[[col for col in test.columns if col not in ['Age', 'Cabin']]] = s_imputer.transform(
            test[[col for col in test.columns if col not in ['Age', 'Cabin']]])
        test.drop(['PassengerId', 'Survived'], axis=1, inplace=True)
        return

    missing_values_(test)

    feature_extraction(test)

    replace_rare(test)

    def categorical_encoding(test_):
        global test

        cat_name = ohe.transform(test_['Name'].to_numpy().reshape(-1, 1)).toarray()
        ohe_df = pd.DataFrame(cat_name, columns=ohe.get_feature_names())
        test_ = pd.concat([test_, ohe_df], axis=1).drop(['Name'], axis=1)

        data__ = col_transformer.transform(test_)
        test = pd.DataFrame(data__, columns=['Sex', 'Embarked'] + [col for col in test_.columns if
                                                                   col not in ['Sex', 'Embarked']])

        return

    categorical_encoding(test)

    def transform_data():
        global test
        test['Fare'], parameters = stats.boxcox(test['Fare'].replace(0, 1))
        test['Fare'], parameters = stats.boxcox(test['Fare'].replace(0, 1))
        return

    transform_data()

    def std():
        global test
        test.Age = std_scaler.transform(test.Age.to_numpy().reshape(-1, 1))
        return

    std()
    return test


logistic_model = LogisticRegression()
logistic_model.fit(data, y_train)
y_pred_logistic = logistic_model.predict(data)


def scoring(y_train, y_pred):
    print('The classification Report:\n ', classification_report(y_train, y_pred))
    print('--------------------------------------------')
    print('The Confusion Matrix:\n ', confusion_matrix(y_train, y_pred))
    return


scoring(y_train, y_pred_logistic)


def score(y_train, y_pred):
    print(classification_report(y_train, y_pred))
    print('----------------------------------------------------------')
    print(confusion_matrix(y_train, y_pred))
    print('----------------------------------------------------------')

    return f1_score(y_train, y_pred)


logistic_score = cross_val_score(logistic_model, data, y_train, scoring=make_scorer(score), cv=5)

svc_model = SVC()
svc_model.fit(data, y_train)
y_pred_scv = svc_model.predict(data)

linearsvc_model = LinearSVC()
linearsvc_model.fit(data, y_train)
y_pred_linearscv = linearsvc_model.predict(data)

scoring(y_train, y_pred_scv)
scoring(y_train, y_pred_linearscv)

svc_score = cross_val_score(svc_model, data, y_train, scoring=make_scorer(score), cv=5)
print('==========================================')
linearsvc_score = cross_val_score(linearsvc_model, data, y_train, scoring=make_scorer(score), cv=5)

forest_model = RandomForestClassifier()
forest_model.fit(data, y_train)
y_pred_forest = forest_model.predict(data)

scoring(y_train, y_pred_forest)
forest_score = cross_val_score(forest_model, data, y_train, scoring=make_scorer(score), cv=5)

kn_model = KNeighborsClassifier()
kn_model.fit(data, y_train)
y_pred_kn = kn_model.predict(data)
scoring(y_train, y_pred_kn)

kn_score = cross_val_score(kn_model, data, y_train, scoring=make_scorer(score), cv=5)
naive_model = GaussianNB()
naive_model.fit(data, y_train)
y_pred_naive = naive_model.predict(data)
scoring(y_train, y_pred_naive)
naive_score = cross_val_score(naive_model, data, y_train, scoring=make_scorer(score), cv=5)

perceptron_model = Perceptron()
perceptron_model.fit(data, y_train)
y_pred_perceptron = perceptron_model.predict(data)

scoring(y_train, y_pred_perceptron)
perceptron_score = cross_val_score(perceptron_model, data, y_train, scoring=make_scorer(score), cv=5)

sgd_model = SGDClassifier()
sgd_model.fit(data, y_train)
y_pred_sgd = sgd_model.predict(data)

scoring(y_train, y_pred_sgd)

sgd_score = cross_val_score(sgd_model, data, y_train, scoring=make_scorer(score), cv=5)

tree_model = DecisionTreeClassifier()
tree_model.fit(data, y_train)
y_pred_tree = tree_model.predict(data)

scoring(y_train, y_pred_tree)
tree_score = cross_val_score(tree_model, data, y_train, scoring=make_scorer(score), cv=5)

logistic_pipeline = Pipeline(
    [('selector', SelectKBest(f_regression)), ('model', LogisticRegression(random_state=42, max_iter=1000))])

logistic_grid = GridSearchCV(estimator=logistic_pipeline, param_grid={'selector__k': [12, 13, 14],
                                                                      'model__C': [0.1, 1, 2],
                                                                      'model__solver': ['liblinear'],
                                                                      'model__penalty': ['l1', 'l2']}, n_jobs=-1,
                             scoring=make_scorer(score), cv=5, verbose=3)

logistic_grid.fit(data, y_train)
print('the best parameters : ', logistic_grid.best_params_)
print('the best score = ', logistic_grid.best_score_)

svc_pipeline = Pipeline([('selector', SelectKBest(f_regression)), ('model', SVC(random_state=42))])

svc_grid = GridSearchCV(estimator=svc_pipeline, param_grid={'selector__k': [12, 13, 14],
                                                            'model__C': [1, 0.5],
                                                            'model__kernel': ['sigmoid', 'linear', 'rbf', 'poly']},
                        n_jobs=-1, scoring=make_scorer(score), cv=5, verbose=3)

svc_pipeline = Pipeline([('selector', SelectKBest(f_regression)), ('model', SVC(random_state=42))])

svc_grid = GridSearchCV(estimator=svc_pipeline, param_grid={'selector__k': [12, 13, 14],
                                                            'model__C': [1, 0.5],
                                                            'model__kernel': ['sigmoid', 'linear', 'rbf', 'poly']},
                        n_jobs=-1, scoring=make_scorer(score), cv=5, verbose=3)

svc_grid.fit(data, y_train)

print('the best parameters : ', svc_grid.best_params_)
print('the best score = ', svc_grid.best_score_)
'''
forest_pipeline = Pipeline(
    [('selector', SelectKBest(f_regression)), ('model', RandomForestClassifier(random_state=42))])

forest_grid = GridSearchCV(estimator=forest_pipeline, param_grid={'selector__k': [11, 12, 13],
                                                                  'model__n_estimators': np.arange(40, 81, 20),
                                                                  'model__max_depth': [5, 7, 9],
                                                                  'model__min_samples_split': [5, 7, 10],
                                                                  'model__max_features': [8, 9, 10]}, n_jobs=-1,
                           scoring=make_scorer(score), cv=7, verbose=3)

forest_grid.fit(data, y_train)
print('the best parameters : ', forest_grid.best_params_)
print('the best score = ', forest_grid.best_score_)

kn_pipeline = Pipeline([('selector', SelectKBest(f_regression)), ('model', KNeighborsClassifier())])

kn_grid = GridSearchCV(estimator=kn_pipeline, param_grid={'selector__k': [12, 13, 14],
                                                          'model__n_neighbors': np.arange(5, 15, 2)}, n_jobs=-1,
                       scoring=make_scorer(score), cv=5, verbose=3)

kn_grid.fit(data, y_train)
print('the best parameters : ', kn_grid.best_params_)
print('the best score = ', kn_grid.best_score_)

tree_pipeline = Pipeline([('selector', SelectKBest(f_regression)), ('model', DecisionTreeClassifier())])

tree_grid = GridSearchCV(estimator=tree_pipeline, param_grid={'selector__k': [10, 11, 12],
                                                              'model__max_depth': [5, 7, 10],
                                                              'model__min_samples_split': [5, 7, 10]}, n_jobs=-1,
                         scoring=make_scorer(score), cv=8, verbose=3)

tree_grid.fit(data, y_train)
print('the best parameters : ', tree_grid.best_params_)
print('the best score = ', tree_grid.best_score_)
'''
##### ------------------------------- test ---------------------------------

test.info()

passengerid = test['PassengerId']
test.drop('PassengerId', axis=1, inplace=True)

processing_pipeline(test)


### ------------------------------------------------------------------------

model = forest_grid.best_estimator_
y_predicted = model.predict(test)
sub = pd.DataFrame()
sub['PassengerId'] = passengerid
sub['Survived'] = y_predicted
sub.to_csv('submission.csv',index=False)


