##Linear regression
# x = independent variables
# y = dependent variables
# y = bo+ b1 * x  =====> b1 egim => coef
#                        bo constant =>bias, intercept_

#### b1 = 


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import data
df = pd.read_csv("/Users/hsekeroglu/Desktop/Huesniye/MLwithPython/Regression/Linear_Regression/example.csv",sep = ",")
print(df.head())

# plot data
#plt.scatter(df.x, df.y)
#plt.xlabel("X")
#plt.ylabel("Y")
#plt.show()

# sklearn library
from sklearn.linear_model import LinearRegression
lineer_regresyon = LinearRegression()


x_value = df.x.values.reshape(-1,1)
y_value = df.y.values.reshape(-1,1)
lineer_regresyon.fit(x_value,y_value)

# prediction
bo = lineer_regresyon.intercept_
b1 = lineer_regresyon.coef_
print(bo)
print(b1)

#Elde edilen regresyon modeli: Y=[4.4701969]+[1.5831968]X

# bo_predict = lineer_regresyon.predict(0)
# print(bo_predict)

y_predicted = lineer_regresyon.predict(x_value)
print(y_predicted)
plt.scatter(df.x , df.y,color = 'red')
plt.plot(df.x , y_predicted, color = 'blue')
plt.show()