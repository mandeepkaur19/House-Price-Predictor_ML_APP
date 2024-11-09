# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:34:49 2024

@author: HP
"""
#Import Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Import dataset

dataset = pd.read_csv(r"D:\NareshIT\Machine Learning\Machine Learning\1. Regression\2. Linear Regression\2.Multi-Linear Regression\Investment.csv")

#Assign X and y to independent and dependent variables

X = dataset.iloc[:, :-1] #Independent variable

y = dataset.iloc[:,4] #Dependent variable


# Convert categorical data (cities) to numerical; ONE HOT ENCODING
X = pd.get_dummies(X, dtype=int)
 

#Split data for train & test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#Linear Regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#Here we predict X_test, because we gave model training data as ref to actual,\
    #so from test we will predict and compare it with training data.
    
#Multi-linear Regression

m = regressor.coef_
print("m: ",m)

print()

c = regressor.intercept_
print("c: ",c)


# Check how many rows in dataset
dataset.shape 


# Add ones column to X

X = np.append(arr = np.ones((50,1)).astype(int), values=X, axis=1)


#Determine which features are significant predictors of the outcome variable.
import statsmodels.api as sm

X_opt = X[:,[0,1,2,3,4,5]]

# Ordinary least square

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()

#remove features IV with p-values greater than a certain threshold, often 0.05
#We eliminate greatest p value (greater than 0.05), run again and do until p<= 0.05



import statsmodels.api as sm

X_opt = X[:,[0,1,2,3,5]]

# Oridinary least square

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()



import statsmodels.api as sm

X_opt = X[:,[0,1,2,3]]

# Oridinary least square

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()



import statsmodels.api as sm

X_opt = X[:,[0,1,3]]

# Oridinary least square

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()



import statsmodels.api as sm

X_opt = X[:,[0,1]]

# Oridinary least square

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()

# here for im getting statisfied p value for 0th and 1st column.
# means in dataset Digital marketing have a significant effect on Profit(dv)


#Training R^2 
train_score = regressor.score(X_train, y_train)
print("Bias: ",train_score)


#Testing R^2
test_score = regressor.score(X_test, y_test)
print("Variance: ",test_score)

#If both bias and variance are high (close to 1), your model is performing wel.
#If bias is high but variance is low, your model might be overfitting.
#if both are low, it may indicate that the model is underfitting.



# Plotting Actual vs. Predicted Values
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, color='blue', label='Test Data Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Perfect Prediction')
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid()
plt.show()

import pickle

filename = 'MLR_model_for_Investments.pkl'
with open(filename, 'wb') as file: pickle.dump(regressor, file)
    
print("Model has been pickled and saved as MLR_model_for_Investments.pkl")

import os
print(os.getcwd())


