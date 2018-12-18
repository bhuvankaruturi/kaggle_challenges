# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 11:31:10 2018

@author: bkarutur
"""
#import required libraries
import numpy as np
import pandas as pd

def getXandY(filePath, mode):
    #read in the data and overview it
    data = pd.read_csv(filePath)
    data.describe()
    
    #store the ids of people in ids and the target variable in y
    ids = data.iloc[:, 0].values
    data.drop(['PassengerId', 'Ticket', 'Name'], inplace=True, axis=1)
    y = None
    if mode == 'train':
        y = data.iloc[:, 0].values
        data.drop(['Survived'], inplace=True, axis=1)
    
    #handle NaN values
    NaNValues = {'Sex': 'male', 'Embarked': 'C'}
    data = data.fillna(value=NaNValues)
    data.Cabin.fillna('0', inplace=True)
    data.loc[:, 'Cabin'] = data.loc[:, 'Cabin'].str[0]
    data.Pclass.fillna(3, inplace=True)
    
    #handle categorical string features
    data.loc[:, 'Pclass'] = data.loc[:, 'Pclass'].astype(str).str[0]
    data = pd.get_dummies(data)
    data.drop(['Pclass_1', 'Sex_female', 'Cabin_0', 'Embarked_C'], inplace=True, axis=1)
    #get the ndarray of features and feature names
    if mode == 'test':
        data.insert(14, 'Cabin_T', 0)
    X = data.iloc[:, :].values
    feature_list = np.array(data.columns)
    return (X, y, ids, feature_list)

def handleMissingValues(X, X_test):
    #fill the missing values with the mean of the column
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    X[:, 0] = imputer.fit_transform(X[:, 0].reshape(-1, 1))[:,0]
    X[:, 3] = imputer.fit_transform(X[:, 3].reshape(-1, 1))[:,0]
    X_test[:, 0] = imputer.transform(X_test[:, 0].reshape(-1, 1))[:,0]
    X_test[:, 3] = imputer.transform(X_test[:, 3].reshape(-1, 1))[:,0]
    return (X, X_test)

def featureScale(X, X_test):
    #feature scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X[:, 0:4] = sc.fit_transform(X[:, 0:4])
    X_test[:, 0:4] = sc.transform(X_test[:, 0:4])
    return (X, X_test)

(X, y, ids, feature_list) = getXandY('train.csv', 'train')
(X_test, y_test, ids_test, feature_list_dummy) = getXandY('test.csv', 'test')

(X, X_test) = handleMissingValues(X, X_test)

#(X, X_test) = featureScale(X, X_test)

#encoding categorical features into colums of binary features
#from sklearn.preprocessing import OneHotEncoder
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()

#delete binary categorical columns to avoid dummy variable trap
#X = np.delete(X, [0, 7, 18], 1)
#feature_list = np.delete(feature_list, [0, 7, 18], 0)

#split the dataset into train and cross validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

#fit the Random Forest Classifier with varying depths
max_score, max_depth_opt = 0, 1
for i in range(1, X.shape[1] + 1):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=1000, max_depth = i, random_state=14)
    clf.fit(X_train, y_train)
    score = clf.score(X_val, y_val)
    if max_score == 0:
        max_score = score
        max_depth_opt = i
    elif max_score < score:
        max_score = score
        max_depth_opt = i
    print("prediction score:", score, " max depth:", i)
    
print("**************************")
print("best score and max_depth", max_score, max_depth_opt)
clf = RandomForestClassifier(n_estimators=500, max_depth = max_depth_opt, random_state=41)
clf.fit(X_train, y_train)

#print the importance of each feature in dataset
importances = list(clf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

y_pred = clf.predict(X_test)
result = np.hstack((ids_test[:, None], y_pred[:, None]))
#np.savetxt('result.csv', result, fmt='%10.0f', delimiter=',')
pd.DataFrame(result, columns = ['PassengerId','Survived']).to_csv('result_rf.csv', index=False)