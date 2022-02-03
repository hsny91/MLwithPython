import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data= pd.read_csv("/Users/hsekeroglu/Desktop/Huesniye/MLwithPython/Classification/SVM/data.csv")
data.drop(["id","Unnamed: 32"],axis=1, inplace=True)

M=data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

plt.scatter(M.radius_mean,M.texture_mean, color="red",label="kotu")
plt.scatter(B.radius_mean,B.texture_mean, color="green",label="iyi")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend() # for showing label
plt.show()

data.diagnosis= [1 if each == "M" else 0 for each in data.diagnosis]

y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)

#Normalization

x= (x_data- np.min(x_data))/(np.max(x_data)-np.min(x_data))

#Train Test Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=(1))


#SVM
from sklearn.svm import SVC
svm = SVC(random_state=1)
svm.fit(x_train,y_train)
print("print accuracy: ", svm.score(x_test,y_test))
# print accuracy:  0.9649122807017544