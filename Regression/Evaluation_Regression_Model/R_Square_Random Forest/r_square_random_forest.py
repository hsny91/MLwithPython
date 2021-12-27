import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/Users/hsekeroglu/Desktop/Udemy/regression/Evaluation_Regression/R_Square_Random_Forest/random_forest_regression_dataset.csv", sep =";", header =None)

x =df.iloc [:,0].values.reshape(-1,1)
y =df.iloc [:,1].values.reshape(-1,1)

#%%  random forest  regression

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100,random_state=42) ## 100 tane desicion tree kullandik

rf.fit(x,y)

y_head = rf.predict(x)


from sklearn.metrics import r2_score

print("r_score ",r2_score(y,y_head))