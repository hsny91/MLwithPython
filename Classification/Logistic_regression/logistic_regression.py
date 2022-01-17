# When we talk about binary classification( 0 and 1 outputs) what comes to mind first is logistic regression.(BINARY CLASSIFICATION)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

#id ve unmaned: 32 delete from table
data.drop(["Unnamed: 32", "id"],axis=1,inplace=True)

#diagnosis M:   B: convert 0 or 1 datatype convert int
data.diagnosis = [1 if each =="M" else 0 for each in data.diagnosis]

print(data.info())

y = data.diagnosis.values   # diagnosis
x_data = data.drop(["diagnosis"],axis= 1) # rest of table except diagnosis

# Normalization: 0-1 scale (x-xmin)/(max(x)-min(x))
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

### train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train shape: ",x_train.shape)