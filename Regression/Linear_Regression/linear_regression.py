import pandas as pd
import matplotlib.pyplot as plt

# import data

df = pd.read_csv("/Users/hsekeroglu/Desktop/Huesniye/MLwithPython/Regression/Linear_Regression/linear_regression_dataset.csv",sep = ";")

# plot data
plt.scatter(df.deneyim, df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

#%% linear regression

# sklearn library
from sklearn.linear_model import LinearRegression

# linear regression model
linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

#%% prediction
import numpy as np

b0 = linear_reg.predict(0)
print("b0: ",b0)

b0_ = linear_reg.intercept_
print("b0_: ",b0_)   # y eksenini kestigi nokta intercept

b1 = linear_reg.coef_
print("b1: ",b1)   # egim slope

# maas = 1663 + 1138*deneyim 

maas_yeni = 1663 + 1138*11
print(maas_yeni)

print(linear_reg.predict(11))

# visualize line
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)  # deneyim

plt.scatter(x,y)
plt.show()

y_head = linear_reg.predict(array)  # maas

plt.plot(array, y_head,color = "red")

linear_reg.predict(100)







#y = maas x = deneyim
# y = bo + b1 * x
# b1 = coeff
# bo = constant(bias)
# maas = bo + b1 * deneyim
# residual = y - y_head(predict value) =error miktari
# n = sample sayisi
# mean squared error
# MSE=sum(residual^2)/n = tüm hatalarin karesini al ve topla, sonra sample böl
# amac min MSE
