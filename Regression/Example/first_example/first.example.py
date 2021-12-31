import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/hsekeroglu/Desktop/Huesniye/MLwithPython/Regression/Example/first_example/column_2C_weka.csv')
#print(data.head(10))

# create data1 that includes pelvic_incidence that is feature and sacral_slope that is target variable
data1 = data[data['class'] =='Abnormal']
print(type(data1)) ## <class 'pandas.core.frame.DataFrame'>

## This orthopedic patients data is not proper for regression so I only use two features that are sacral_slope and pelvic_incidence of abnormal
x = np.array(data1.loc[:,'pelvic_incidence']).reshape(-1,1)
y = np.array(data1.loc[:,'sacral_slope']).reshape(-1,1)

print(type(y)) ## <class 'numpy.ndarray'>

# Scatter
plt.scatter(x,y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()