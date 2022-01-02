import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/hsekeroglu/Desktop/Huesniye/MLwithPython/Regression/Examples/linear_regression/column_2C_weka.csv')
#print(data.head(10))

# create data1 that includes pelvic_incidence that is feature and sacral_slope that is target variable
data1 = data[data['class'] =='Abnormal']
print(type(data1)) ## <class 'pandas.core.frame.DataFrame'>

## This orthopedic patients data is not proper for regression so I only use two features that are sacral_slope and pelvic_incidence of abnormal
x = np.array(data1.loc[:,'pelvic_incidence']).reshape(-1,1)
y = np.array(data1.loc[:,'sacral_slope']).reshape(-1,1)

print(type(y)) ## <class 'numpy.ndarray'>


# LinearRegression
from sklearn.linear_model import LinearRegression

# linear regression model
linear_reg = LinearRegression()

# Fit
linear_reg.fit(x,y)

##Predict
# y_head = bo+ b1 * x
bo= linear_reg.intercept_
b1 = linear_reg.coef_
# print(bo,b1) #[2.17390961] [[0.66047069]]
predict_sacrol_scope = bo + b1*63
predict_sacrol_scope_= linear_reg.predict(np.array(63).reshape(-1,1))
print(predict_sacrol_scope) #[[43.78356298]]
print(predict_sacrol_scope_) #[[43.78356298]]

#Plot
y_head = linear_reg.predict(x)
# Scatter
plt.scatter(x,y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.plot(x,y_head,color="red")
plt.show()


##### Evaluation_Regression ########
from sklearn.metrics import r2_score
print("r2_score: ", r2_score(y,y_head))  ### r2_score:  0.6458410481075871