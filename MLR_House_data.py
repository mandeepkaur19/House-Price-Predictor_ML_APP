# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:21:12 2024

@author: HP
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"D:\NareshIT\Machine Learning\1. Regression\2. Linear Regression\2.Multi-Linear Regression\House_data.csv")

X = pd.concat([dataset.iloc[:,3:16], dataset.iloc[:,19]], axis=1)

y = dataset.iloc[:,2]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


m = regressor.coef_
print("m: ",m)


c = regressor.intercept_
print("c: ",c)


dataset.shape

X = np.append(arr = np.ones((21613,1)).astype(int), values=X, axis=1)

import statsmodels.api as sm

X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]] 

regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()

regressor_ols.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,5,6,7,8,9,10,11,12,13,14]] 
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
regressor_ols.summary()

train_score = regressor.score(X_train, y_train)
print('traning score: ', train_score)
print("Bias: ",train_score*100)

test_score = regressor.score(X_test, y_test)
print('testing score: ', test_score)
print("Variance: ",test_score*100)

# bias and variance are almost equal means its good fit.

plt.figure(figsize=(12,6))
plt.scatter(y_test, y_pred, color='blue', label='Test Data Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Perfect Prediction')
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid()
plt.show()











