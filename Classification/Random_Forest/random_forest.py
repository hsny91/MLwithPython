import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data= pd.read_csv("/Users/hsekeroglu/Desktop/Huesniye/MLwithPython/Classification/Random_Forest/data.csv")

data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

#from sklearn import preprocessing
#le_diagnosis=preprocessing.LabelEncoder()
#le_diagnosis.fit(["B","M"])
#data.iloc[:,1]=le_diagnosis.transform(data.iloc[:,1])

data.diagnosis=[1 if each =="M" else 0 for each in data.diagnosis]
y= data.diagnosis.values
x_data=data.drop(["diagnosis"], axis=1)

# Normalization
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.15, random_state=42)


#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=1) ##100 desicion Tree
rf.fit(x_train,y_train)
print("random forest algo result: ", rf.score(x_test,y_test))

y_pred= rf.predict(x_test)
y_true= y_test

#Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)

# visulaton
import seaborn as sns
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linecolor="red", fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()