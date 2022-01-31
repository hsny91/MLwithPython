import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/hsekeroglu/Desktop/Huesniye/MLwithPython/Regression/Polynomial_Linear_Regression/polynomial+regression.csv", sep =";")

y =df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)

# plt.scatter(x,y)
# plt.ylabel("araba_max_hiz" )
# plt.xlabel("araba_fiyat") 
# plt.show()


# linear regression  y = bo+ b1*x
# multiple linear regression  y =bo+ b1*x1 + b2*x2.........+ bn*xn

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)
y_head = lr.predict(x)
plt.plot(x,y_head, color = "red")
plt.show()
print(lr.predict(10000))

#polynominal regression y = bo+b1*x+ b2*x^2 + ....... bn* x^n

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression  = PolynomialFeatures(degree =2) # degree artarsa polinom degeri artar daha dogru sonuclar elde edebiliriz.
x_polynominal =polynomial_regression.fit_transform(x)

linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynominal,y)
y_head2 = linear_regression2.predict(x_polynominal)
plt.plot(x,y_head2, color = "green",label ="poly")
plt.legend()
plt.show()