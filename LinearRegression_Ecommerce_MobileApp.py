# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 16:57:36 2022

@author: avidvans3
"""
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats







data=pd.read_csv('Ecommerce_Customers.csv')

# data=data.drop('Index',axis=1)

# data=data[data['Avg. Session Length'].str.contains("nan")==False]

data=pd.DataFrame(data=data)

df = data.apply (pd.to_numeric, errors='coerce')

df.dropna(how='all')

df.drop('Email', axis=1, inplace=True)
df.drop('Address', axis=1, inplace=True)

df.drop('Avatar', axis=1, inplace=True)

df = df.dropna()



Y=df['Yearly Amount Spent']
df.drop('Yearly Amount Spent', axis=1, inplace=True)
X=df;

X = sm.add_constant(X)
est = sm.OLS(Y, X)
est2 = est.fit()
print(est2.summary())








