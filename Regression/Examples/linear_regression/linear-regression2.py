from turtle import color
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv("/Users/hsekeroglu/Desktop/Huesniye/MLwithPython/Regression/Examples/linear_regression/FuelConsumptionCo2.csv",sep = ",")
#print(df.head())


# summarize the data
#print(df.describe())

#Let's select some features to explore more.
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
#print(cdf.head(9))

#We can plot each of these features:
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
#viz.hist()
#plt.show()

#Now, let's plot each of these features against the Emission, to see how linear their relationship is:

# plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("Emission")
# plt.show()

# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()


from sklearn import linear_model
linear_reg = linear_model.LinearRegression()
x = df.ENGINESIZE.values.reshape(-1,1)
y = df.CO2EMISSIONS.values.reshape(-1,1)
linear_reg.fit(x,y)


b0_ = linear_reg.intercept_
print("b0_: ",b0_)   # y eksenini kestigi nokta intercept #125.3040995]

b1 = linear_reg.coef_
print("b1: ",b1)   # egim slope #slope 39.12519979

# y_head = bo+b1.x1
y_head_1 = 125+39*2.4
y_head_2 = linear_reg.predict(np.array(2.4).reshape(-1,1))
print(y_head_1,y_head_2)
y_predicted = linear_reg.predict(x)

# visualize line
plt.scatter(x,y)
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.plot(x,y_predicted ,color="red")
plt.show()

#  Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
from sklearn.metrics import r2_score

print("r_score: ", r2_score(y,y_predicted )) # r_score:  0.7641458597854816