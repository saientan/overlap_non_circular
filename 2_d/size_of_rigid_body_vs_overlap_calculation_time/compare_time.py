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
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

import time


#########################################################################
N=64    
#N=30
#phi=0.69#number_density
phi=0.69    
L=(N/phi)**0.5
#print (L)

################list_of_classifier####################

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
    "Grad Boost",
    "HistGradientBoostingClassifier",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    GradientBoostingClassifier(),
    HistGradientBoostingClassifier()
]

#######################################################

#########################The_Extended_Object##################################
from numpy.random import seed
seed(1)


data_object=pylab.loadtxt('coord.dat')
x_object=data_object[:,0]
y_object=data_object[:,1]
length=len(x_object)

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


####################Extended_Object_Overlap_Detection##################


def overlap(x,y,theta,xnew,ynew,thetanew,L):
    dx=abs(x-xnew)
    dy=abs(y-ynew)
    if dx>L/2:
        dx=L-dx
    if dy>L/2:
        dy=L-dy
    overlap=0
    diff_theta=theta-thetanew
    if diff_theta<0:
        diff_theta=2*np.pi-abs(diff_theta)


    x_object_new, y_object_new=rotate(x_object,y_object,diff_theta)
    x_object_new, y_object_new=translate(x_object_new, y_object_new,dx,dy)
    

    for i in range(0,len(x_object)):
        for j in range(0,len(x_object_new)):
            d=((x_object[i]-x_object_new[j])**2+(y_object[i]-y_object_new[j])**2)**0.5
            if d<=0.11111111111111116: #######Which Cutoff Between the Particles########
                overlap=1 
                break
    return overlap

###################Machine_Learning_Overlap_Detection##########################

def overlap_ML(x,y,theta,xnew,ynew,thetanew,model,L):

    #dx=abs(x-xnew)
    dx=x-xnew
    #dy=abs(y-ynew)
    dy=y-ynew
    if dx>L/2:
        dx=L-dx
    if dy>L/2:
        dy=L-dy

    if dx<-L/2:
        dx=-L-dx
    if dy<-L/2:
        dy=-L-dy
 
    diff_theta=theta-thetanew
    #if diff_theta<0:
    #    diff_theta=2*np.pi-abs(diff_theta)
    ml_in=[[dx, dy, diff_theta]]
    #ml_in=ml_in.reshape(1, -1)
    ov_ml=model.predict(ml_in)
    #if ov_ml<0.5:
    #    ov_ml=0
    #if ov_ml>=0.5:
    #    ov_ml=1
    return ov_ml


#data_ML=pylab.loadtxt('training_data.dat')
data_ML=pylab.loadtxt('training_data.dat')
#x_train=data_ML[:,0:3]
#y_train=data_ML[:,3]

xml=data_ML[:,0:3]
yml=data_ML[:,3]
x_train, x_test, y_train, y_test = train_test_split(xml, yml, test_size=0.1 ,random_state=0)
#model =  RandomForestClassifier(class_weight="balanced")
#model =  RandomForestClassifier()


for something in range (0,len(names)):
    #print (names[something])
    start= (time.time())

    model=classifiers[something]


    #model = GradientBoostingClassifier()

    model.fit(x_train,y_train.ravel())
    pred_train=model.predict(x_train)
    accuracy=0
    for i in range(0,len(x_train)):
        if y_train[i]==pred_train[i]:
            accuracy=accuracy+1
    #print ("train_accuracy", accuracy/len(x_train))
    pred_test=model.predict(x_test)
    accuracy=0
    for i in range(0,len(x_test)):
        if y_test[i]==pred_test[i]:
            accuracy=accuracy+1
#print ("test_accuracy", accuracy/len(x_test))

#############################################


    newx=random.uniform(0,L)
    newy=random.uniform(0,L)
    newtheta=random.uniform(0,2*np.pi)

    oldx=random.uniform(0,L)
    oldy=random.uniform(0,L)
    oldtheta=random.uniform(0,2*np.pi)



    start_NORMAL = time.time()
    o_lap=overlap(newx,newy,newtheta,oldx,oldy,oldtheta,L)
    end_NORMAL = time.time()

    start_ML = time.time()
    o_lap=overlap_ML(newx,newy,newtheta,oldx,oldy,oldtheta,model, L)
    end_ML = time.time()


    print (names[something],length**2, end_NORMAL-start_NORMAL, end_ML-start_ML)







