import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("/Users/hsekeroglu/Desktop/Huesniye/MLwithPython/Classification/K-Nearest_Neigbour/data.csv")

data.drop(["id","Unnamed: 32"],axis=1, inplace=True)
data.head()
# malignant = M =>> KOTU HUYLU
# Benign = B >> IYI HUYLU

M = data[data.diagnosis== "M"]
B = data[data.diagnosis== "B"]
print(B.info())

# Scatter plot
plt.scatter(M.radius_mean,M.area_mean,color = "red", label="kotu")
plt.scatter(B.radius_mean,B.area_mean,color = "green", label="iyi")
plt.xlabel("radius_mean")
plt.ylabel("area_mean")
plt.legend() # show label
plt.show()

# Scatter plot
plt.scatter(M.radius_mean,M.texture_mean,color = "red", label="kotu")
plt.scatter(B.radius_mean,B.texture_mean,color = "green", label="iyi")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend() # show label
plt.show()

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data= data.drop(["diagnosis"],axis=1)

# Normalization
x = (x_data - np.min(x_data))/(np.max(x_data)- np.min(x_data))

# Train test split
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)

#KNN Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3) # komsu 3 degeri al
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print(prediction)
print(" {}  nn score: {}".format(3,knn.score(x_test,y_test)))

# find k value
score_list=[]
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
plt.plot(range(1,15),score_list) # best k value 8
plt.xlabel=("k values")
plt.ylabel = (" accuracy")
plt.show()