# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:59:02 2018

@author: bkarutur
"""
#import necessary libraries
import numpy as np
import pandas as pd

def preprocessing(location, mode):
    #read the training data
    data =pd.read_csv(location)
    data.drop(['Ticket', 'Name'],inplace =True, axis=1)
    
    # Taking care of missing data in string features
    NaNValues = {'Sex': 'female', 'Embarked': 'S'}
    data = data.fillna(value=NaNValues)
    
    #Handling missing values in Cabin
    data.Cabin.fillna('0', inplace=True)
    data.loc[:, 'Cabin'] = data.loc[:, 'Cabin'].str[0]
    
    x_values = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    if mode == 'test':
        x_values = x_values - 1
    if mode == 'train':
        y = data.iloc[:, 1].values
    
    X = data.iloc[:, x_values].values
    ids = data.iloc[:, 0].values
    
    #Handling NaN in numeric features
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    X[:, 2] = imputer.fit_transform(X[:, 2].reshape(-1, 1))[:,0]
    X[:, 5] = imputer.fit_transform(X[:, 5].reshape(-1, 1))[:,0]
    
    if mode == 'test':
        return (X, ids)
    return (X, y, ids)

def encodeCategoricalFeatures(X, X_test):    
    #Encoding categorical variables
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    labelencoder_X = LabelEncoder()
    
    #training set
    X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
    X[:, -2] = labelencoder_X.fit_transform(X[:, -2])
    X[:, -1] = labelencoder_X.fit_transform(X[:, -1])
    
    #test set
    X_test[:, 1] = labelencoder_X.fit_transform(X_test[:, 1])
    X_test[:, -2] = labelencoder_X.fit_transform(X_test[:, -2])
    X_test[:, -1] = labelencoder_X.fit_transform(X_test[:, -1])
    
    #Encoding categorical variables
    onehotencoder = OneHotEncoder(categorical_features = [0, -2, -1])
    X = onehotencoder.fit_transform(X).toarray()
    X_test = onehotencoder.transform(X_test).toarray()
    
    #remove one column of categorical variable to avoid dummy variable trap
    X = np.delete(X, [0, 3, 6], 1)
    X_test = np.delete(X_test, [0, 3, 6], 1)
    
    return(X, X_test)

def fitModel(X_train, y_train):
    #feature scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train[:, 5:8] = sc_X.fit_transform(X_train[:, 5:8])
    
    #fit the logistic regression model and then make the predictions
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 2, solver='liblinear', multi_class='ovr').fit(X_train, y_train);
    return classifier

def splitData(X, y):
    #split the data in training and cross validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    return (X_train, X_test, y_train, y_test)

(X, y, ids) = preprocessing('train.csv', 'train')
(X_test, ids_test) = preprocessing('test.csv', 'test')

(X, X_test) = encodeCategoricalFeatures(X, X_test)
(X_train, X_val, y_train, y_val) = splitData(X, y)
classifier = fitModel(X_train, y_train)

#cross validation
#accuracy = classifier.score(X_val, y_val)
#print(accuracy)

#test set result
y_pred = classifier.predict(X_test)

