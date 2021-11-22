import pandas as pd
import matplotlib.pyplot as plt

# import data

df = pd.read_csv("/Users/hsekeroglu/Desktop/Huesniye/MLwithPython/Regression/Linear_Regression/linear_regression_dataset.csv",sep = ";")

# plot data
plt.scatter(df.deneyim, df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

#%linear regression

# # sklearn library
# from sklearn.linear_model import LinearRegression

# # linear regression model
# linear_reg = LinearRegression()




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
