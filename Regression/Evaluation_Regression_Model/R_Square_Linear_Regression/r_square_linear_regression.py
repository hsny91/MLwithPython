import pandas as pd
import matplotlib.pyplot as plt

# import data

df = pd.read_csv("linear_regression_dataset.csv",sep = ";")

from sklearn.linear_model import LinearRegression

# linear regression model
linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)
linear_reg.fit(x,y)
y_head = linear_reg.predict(x)

# visualize line

plt.scatter(x,y)
plt.plot(x, y_head,color = "red")
plt.show()

from sklearn.metrics import r2_score
print("r_score: ", r2_score(y,y_head))

