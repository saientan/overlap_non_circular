from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random
import pylab
import os
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import sklearn.linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

N=400
a=[]
b=[]
for i in range(1,N):
    j=100*i
    s="head -"+str(j)+" new_training_data.dat > learning_data.dat"
    os.system(s)
    

    data_ML=pylab.loadtxt('learning_data.dat')
    x_train=data_ML[:,0:3]
    y_train=data_ML[:,3]
    print (x_train)
    #y_train=y_train.reshape(-1, 1)
    data_ML=pylab.loadtxt('testing_data.dat')
    x_test=data_ML[:,0:3]
    y_test=data_ML[:,3]
    y_test=y_test.reshape(-1, 1) 
    #model = GradientBoostingClassifier()
    #model=GradientBoostingClassifier(n_estimators=50, max_depth=3)
    #model = RandomForestClassifier()
    model= model=KNeighborsClassifier(3)
    model.fit(x_train,y_train)

    pred_y_test=model.predict(x_test)

    accuracy=0
    for k in range(0,len(y_test)):
        if y_test[k]==pred_y_test[k]:
            accuracy=accuracy+1
    accuracy=(accuracy*100)/(len(y_test))
    a.append(j)
    b.append(accuracy)

plt.xscale('log',base=10)
plt.scatter(a,b)
plt.plot(a,b)
plt.show()




