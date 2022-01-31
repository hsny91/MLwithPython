import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df= pd.read_csv("teleCust1000t.csv")

# The target field, called custcat, has four possible values that correspond to the four customer groups, as follows: 1- Basic Service 2- E-Service 3- Plus Service 4- Total Service
# Let’s see how many of each class is in our data set¶
print(df['custcat'].value_counts()) 
y = df.custcat.values
x_data= df.drop(["custcat"], axis=1)

#3    281
#1    266
#4    236
#2    217
#Name: custcat, dtype: int64

df.hist(column='income', bins=50) # hist: A histogram, bins: kutu sayisi

# Normalize Data
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)) 


#Train test Split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=(4))

#KNN MODEL
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train,y_train)

prediction = knn.predict(x_test)
print(prediction)
print(" {}  nn score: {}".format(3,knn.score(x_test,y_test))) #  3  nn score: 0.315

# find K value
score_list = []
for each in range (1,15):
    knn2= KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
plt.plot(range(1,15),score_list)
plt.xlabel=("k Value")
plt.ylabel(" accuracy")
plt.show() # best value k = 8,10