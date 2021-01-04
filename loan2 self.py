# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:09:38 2020

@author: Abhishek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
df = pd.read_csv('customer_churn.csv')
df.head()
df.columns
df.isnull().sum()
df.duplicated().sum()
df.info()
df.dtypes
df.describe()
# change data type
df['TotalCharges']  = pd.to_numeric(df['TotalCharges'],errors='coerce')
df.corr()
#checking null values
df1 = df[df.TotalCharges.isnull()]
df1 
#droping na
df2 = df.dropna(axis=0,how='any')
df2
#OUTLIER sinior citizen
figure = plt.figure(figsize=(10,7))
plt.boxplot(df2['SeniorCitizen'])
# tenure
figure = plt.figure(figsize=(10,7))
plt.boxplot(df2['tenure'])
#monthlycharges
figure = plt.figure(figsize=(10,7))
plt.boxplot(df2['MonthlyCharges'])
#total charges
figure = plt.figure(figsize=(10,7))
plt.boxplot(df2['TotalCharges'])
#dummies
df2.drop(['customerID','gender' ], axis = 1, inplace = True) 

df3= pd.get_dummies(df2,drop_first=True)

df3.head(20)


data = df3.copy()
df3.info()

X = df3.drop(['Churn_Yes'],1)
Y = df3['Churn_Yes']

X = np.array(X)
Y = np.array(Y)


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
#decision treee

clf_DT = DecisionTreeClassifier()
clf_KNN = KNeighborsClassifier()
#split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3)

#fittinf data

clf_DT.fit(X_train, Y_train) #.fit() to fit the data for learning
pred_DT = clf_DT.predict(X_test) # Predicting the Taregt Class
score_DT = (clf_DT.score(X_test,Y_test)*100) # Calculating the Accuracy
# print(pred_DT)
print('Accuracy DT','\t',np.round(score_DT,3))
target_names = ['class 0', 'class 1']

print('\n\n','The Confussion Metric is:\n',confusion_matrix(Y_test,pred_DT))

print('\n\n','The Classification report is:\n',classification_report(Y_test,pred_DT,target_names=target_names))

# knn
clf_KNN.fit(X_train, Y_train) #.fit() to fit the data for learning
pred_KNN = clf_KNN.predict(X_test) # Predicting the Taregt Class
score_KNN = (clf_KNN.score(X_test,Y_test)*100) # Calculating the Accuracy
print('Accuracy KNN','\t',np.round(score_KNN,3))

target_names = ['class 0', 'class 1']

print('\n\n','The Confussion Metric is:\n',confusion_matrix(Y_test,pred_KNN))

print('\n\n','The Classification report is:\n',classification_report(Y_test,pred_KNN,target_names=target_names))


