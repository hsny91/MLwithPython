import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


x1 = np.random.normal(25,5,1000)
y1 = np.random.normal(25,5,1000)

x2 = np.random.normal(55,5,1000)
y2 = np.random.normal(60,5,1000)

x3 = np.random.normal(55,5,1000)
y3 = np.random.normal(15,5,1000)


x = np.concatenate((x1,x2,x3),axis=0)
y = np.concatenate((y1,y2,y3),axis=0)

dictionary ={"x":x,"y":y}
data = pd.DataFrame(dictionary)

print(data.describe())

#Visulazition
plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.show()

# Dendrogram
from scipy.cluster.hierarchy import linkage, dendrogram
merg = linkage(data, method = "ward")
dendrogram (merg, leaf_rotation=90)
plt.xlabel('data points')
plt.ylabel("euclidean distance")
plt.show()


# Hierarchical Cluster

from sklearn.cluster import AgglomerativeClustering
hierarchical_cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage="ward")
cluster = hierarchical_cluster.fit_predict(data)
data['label']=cluster

#Visualization
plt.scatter(data.x[data.label==0] , data.y[data.label==0], color="red")
plt.scatter(data.x[data.label ==1], data.y[data.label==1], color="green")
plt.scatter(data.x[data.label==2], data.y[data.label==2],color="blue")
plt.show()