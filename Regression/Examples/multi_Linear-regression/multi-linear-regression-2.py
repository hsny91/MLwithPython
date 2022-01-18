import imp
from re import X
from statistics import linear_regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df= pd.read_csv("/Users/hsekeroglu/Desktop/Huesniye/MLwithPython/Regression/Examples/multi_Linear-regression/FuelConsumptionCo2.csv")



from sklearn.linear_model import LinearRegression
multiLinear_reg= LinearRegression()
y=df.CO2EMISSIONS.values.reshape(-1,1)
x = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]
multiLinear_reg.fit(x,y)
print ('Coefficients: ', multiLinear_reg.coef_) # Coefficients:  [[10.85524041  7.51622501  9.59563161]]