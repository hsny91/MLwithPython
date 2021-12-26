
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


df = pd.read_csv("/Users/hsekeroglu/Desktop/Huesniye/MLwithPython/Regression/Multiple_Linear_Regression",sep = ";")
# linear regression model


y = df.maas.values.reshape(-1,1)
x = df.iloc[:,[0,2]].values
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)
print("bo_", multiple_linear_regression.intercept_)
print("b1,b2", multiple_linear_regression.coef_)
multiple_linear_regression.predict(np.array([[10,35],[5,35]]))

# https://medium.com/@afozbek_/sklearn-k%C3%BCt%C3%BCphanesi-kullanarak-linear-regression-modeli-nas%C4%B1l-geli%C5%9Ftirilir-692a0bf13998