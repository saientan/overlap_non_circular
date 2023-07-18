import pylab
import numpy as np
import matplotlib.pyplot as plt

cutoff=0.11111111111111116

data=pylab.loadtxt('coord.dat')
x=data[:,0]
y=data[:,1]

f=open('training_data.dat','w')

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
    print(len(x),len(xnew))

def cal_overlap(x,y,xnew,ynew):
    overlap=0
    for i in range(0,len(x)):
        for j in range(0,len(xnew)):
            d=((x[i]-xnew[j])**2+(y[i]-ynew[j])**2)**0.5
            if d<=cutoff:
                overlap=1
                break
    return overlap

def cal_energy(x,y,xnew,ynew):
    energy=0
    for i in range(0,len(x)):
        for j in range(0,len(xnew)):
            d=((x[i]-xnew[j])**2+(y[i]-ynew[j])**2)**0.5
            energy=energy+(((1/d)**12)-((1/d)**6))
            
    return energy



for i in range(0,10000):
    #print (i)
    theta1=np.random.uniform(0,2*np.pi)
    xnew, ynew=rotate(x,y,theta1)
    dist_x1=np.random.uniform(0,1)
    dist_y1=np.random.uniform(0,1)
    xnew1, ynew1=translate(xnew, ynew,dist_x1,dist_y1)

    #energy=cal_energy(x,y,xnew,ynew)
    #overlap=cal_overlap(x,y,xnew,ynew)
    
    theta2=np.random.uniform(0,2*np.pi)
    xnew, ynew=rotate(x,y,theta2)
    dist_x2=np.random.uniform(0,10)
    dist_y2=np.random.uniform(0,10)
    xnew2, ynew2=translate(xnew, ynew,dist_x2,dist_y2)

    overlap=cal_overlap(xnew1,ynew1,xnew2,ynew2)

    diff_x=dist_x1-dist_x2
    diff_y=dist_y1-dist_y2
    diff_theta=theta1-theta2

    
    if overlap==0:
        f.write(str(diff_x)+'   '+str(diff_y)+'   '+str(diff_theta)+'   '+str(overlap)+'\n')
        f.write(str(-diff_x)+'   '+str(-diff_y)+'   '+str(-diff_theta)+'   '+str(overlap)+'\n')
    if overlap==1:
        f.write(str(diff_x)+'   '+str(diff_y)+'   '+str(diff_theta)+'   '+str(overlap)+'\n') 
        f.write(str(-diff_x)+'   '+str(-diff_y)+'   '+str(-diff_theta)+'   '+str(overlap)+'\n')

    
    


