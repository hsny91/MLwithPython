import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import  skew


df = pd.read_csv("/Users/hsekeroglu/Desktop/Huesniye/MLwithPython/Regression/Examples/multi_Linear-regression/Advertising.csv",sep=",")
print(df.head())
sns.pairplot(df,x_vars=['TV','radio','newspaper'],y_vars='sales',height=7,aspect=0.7)
#MultiLinear Regression
from sklearn.linear_model import LinearRegression
multi_regression = LinearRegression() 
x = df[['TV','radio','newspaper']] #x = df.iloc[:,[0,1,2]].values
y = df.sales  ##y = df.sales.values.reshape(-1,1)
multi_regression.fit(x,y)

print(multi_regression.intercept_)
print(multi_regression.coef_)

list(zip(['TV','radio','newspaper'],multi_regression.coef_))

#sns.heatmap(df.corr(), annot= "True")