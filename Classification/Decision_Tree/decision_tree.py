import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data= pd.read_csv("/Users/hsekeroglu/Desktop/Udemy/Classification/Decision_Tree/data.csv")

data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
data.diagnosis=[1 if each =="M" else 0 for each in data.diagnosis]
y= data.diagnosis.values
x_data=data.drop(["diagnosis"], axis=1)

# Normalization
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.15, random_state=42)

#Decision Tree Algoritma

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("score: ", dt.score(x_test, y_test))