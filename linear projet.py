# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 18:30:41 2020

@author: Abhishek
"""
#importing library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Reading data from remote link

url = "http://bit.ly/w-data"
student_data = pd.read_csv(url)
print("Data imported successfully")
student_data.head(10)
student_data.info()
student_data.describe()
#Plotting the distribution of scores
student_data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hour Studied')
plt.ylabel('Percentage Score')
plt.show()
#devide into attributes
X = student_data.iloc[:, :-1].values
y = student_data.iloc[:, 1].values
# split the data into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                           test_size=0.2, random_state=0)
# linear model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print("Training complete.")

# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_
# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()
# check prediction
print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores

# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df

# You can also test with your own data
hours =np.array([9.25]) 
hours = hours.reshape(-1,1)
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))

#evaluting the model
from sklearn import metrics
print('Mean Absolute Error:',
metrics.mean_absolute_error(y_test, y_pred))






