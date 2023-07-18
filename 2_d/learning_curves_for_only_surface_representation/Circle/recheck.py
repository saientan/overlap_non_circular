import pylab
import numpy as np
import matplotlib.pyplot as plt



cutoff=0.11111111111111116

data=pylab.loadtxt('coord.dat')
x=data[:,0]
y=data[:,1]

data=pylab.loadtxt('training_data.dat')
x_train=data[:,0]
y_train=data[:,1]
theta_train=data[:,2]
overlap_train=data[:,3]

def translate(x,y,dist_x,dist_y):
    xnew=x+dist_x
    ynew=y+dist_y
    return xnew, ynew

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


def cal_overlap(x,y,xnew,ynew):
    overlap=0
    for i in range(0,len(x)):
        for j in range(0,len(xnew)):
            d=((x[i]-xnew[j])**2+(y[i]-ynew[j])**2)**0.5
            if d<=cutoff:
                overlap=1
                break
    return overlap


for i in range(0,len(x_train)):
    xnew,ynew=rotate(x,y,theta_train[i])
    xnew,ynew=translate(xnew,ynew,x_train[i],y_train[i])
    d=[]
    for j in range(0,len(x)):
        for k in range(0,len(xnew)):
            dist=((x[j]-xnew[k])**2+(y[j]-ynew[k])**2)**0.5
            d.append(dist)    
    
    print (min(d),overlap_train[i])


