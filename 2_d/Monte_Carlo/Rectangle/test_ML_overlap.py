from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random
import pylab

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

import pylab
import numpy as np

#########################Translation##################

def translate(x,y,dist_x,dist_y):
    xnew=x+dist_x
    ynew=y+dist_y
    return xnew, ynew

####################Rotation##################

def rotate(x,y,theta):
    cmx=sum(x)/len(x)
    cmy=sum(y)/len(y)

    xr=x-cmx
    yr=y-cmy
    xrnew=((xr*np.cos(theta))-(yr*np.sin(theta)))
    yrnew=((xr*np.sin(theta))+(yr*np.cos(theta)))
    xnew=xrnew+cmx
    ynew=yrnew+cmy
    return xnew, ynew




data=pylab.loadtxt('x_y_theta')
x=data[:,0]
y=data[:,1]
theta=data[:,2]

data_ML=pylab.loadtxt('training_data.dat')
#x_train=data_ML[:,0:3]
#y_train=data_ML[:,3]

xml=data_ML[:,0:3]
yml=data_ML[:,3]
x_train, x_test, y_train, y_test = train_test_split(xml, yml, test_size=0.1 ,random_state=0)
#model =  RandomForestClassifier(class_weight="balanced")
#model =  RandomForestClassifier()

model = GradientBoostingClassifier()

model.fit(x_train,y_train.ravel())


for i in range(0,len(x)):
    for j in range(i+1,len(x)):
        dx=x[i]-x[j]
        dy=y[i]-y[j]
        diff_theta=theta[i]-theta[j]
        ml_in=[[dx, dy, diff_theta]]
        ov_ml=model.predict(ml_in)
        if ov_ml==1:
            print (i, j, ov_ml)
 


data_object=pylab.loadtxt('coord_rectangle.dat')
x_object=data_object[:,0]
y_object=data_object[:,1]



initial_snap=pylab.loadtxt('initial_snap.dat')
x=data[:,0]
y=data[:,1]
theta=data[:,2]

extended_x=[]
extended_y=[]

for i in range(0,len(x)):
    plt.scatter(x[i], y[i],label=str(i))
    plt.text(x[i], y[i],str(i))

#plt.legend()
#plt.show()

for i in range(0,len(x)):
    x_object_n, y_object_n=rotate(x_object,y_object,theta[i])
    x_object_n, y_object_n=translate(x_object_n, y_object_n,x[i],y[i])
    for j in range(0,len(x_object_n)):
        extended_x.append(x_object_n[j])
        extended_y.append(y_object_n[j])
plt.scatter(extended_x,extended_y, color='red', alpha=0.1)
plt.show()












