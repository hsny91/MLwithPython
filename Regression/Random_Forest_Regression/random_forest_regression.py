## ensemble learning: ayni aynda bircok algoritmayi kullanmak
## random forest: agaclarin toplami örnek:film tavsiye

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/Users/hsekeroglu/Desktop/Huesniye/MLwithPython/Regression/Random_Forest_Regression/random_forest_regression_dataset.csv", sep =";", header =None)

x =df.iloc [:,0].values.reshape(-1,1)
y =df.iloc [:,1].values.reshape(-1,1)

#%%  random forest  regression

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100,random_state=42) ## 100 tane desicion tree kullandik

rf.fit(x,y)

print(rf.predict(np.array(7.8).reshape(-1,1)))
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x_)


###visualize

plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color = "green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()