from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random
import pylab
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#import sklearn
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import r2_score
#import sklearn.linear_model
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.neural_network import MLPRegressor
#from sklearn.neural_network import MLPClassifier
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn.ensemble import HistGradientBoostingClassifier
import time

#########################Translation##################

def translate(x,y,dist_x,dist_y):
    xnew=x+dist_x
    ynew=y+dist_y
    return xnew, ynew

def translate_3d(x,y,z,dist_x,dist_y,dist_z):
    xnew=x+dist_x
    ynew=y+dist_y
    znew=z+dist_z
    return xnew, ynew, znew



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

def rotate_3d(x,y,z,alpha,beta,gamma):
    cmx=sum(x)/len(x)
    cmy=sum(y)/len(y)
    cmz=sum(z)/len(z)

    xr=x-cmx
    yr=y-cmy
    zr=z-cmz

    xrnew=(xr*np.cos(beta)*np.cos(gamma))+(yr*((np.sin(alpha)*np.sin(beta)*np.cos(gamma))-(np.cos(alpha)*np.sin(gamma))))+(zr*((np.cos(alpha)*np.sin(beta)*np.cos(gamma))+(np.sin(alpha)*np.sin(gamma))))

    yrnew=(xr*np.cos(beta)*np.sin(gamma))+(yr*((np.sin(alpha)*np.sin(beta)*np.sin(gamma))+(np.cos(alpha)*np.cos(gamma))))+(zr*((np.cos(alpha)*np.sin(beta)*np.sin(gamma))-(np.sin(alpha)*np.cos(gamma))))

    zrnew=(-xr*np.sin(beta))+(yr*np.sin(alpha)*np.cos(beta))+(zr*np.cos(alpha)*np.cos(beta))

    xnew=xrnew+cmx
    ynew=yrnew+cmy
    znew=zrnew+cmz

    return xnew, ynew, znew

data_object=pylab.loadtxt('coord.dat')
x_object=data_object[:,0]
y_object=data_object[:,1]
z_object=data_object[:,2]

#full_object=pylab.loadtxt('full_star_coord.dat')
#x_full_object=full_object[:,0]
#y_full_object=full_object[:,1]

g=open('for_vmd_ML.xyz','w')
#L=6.23  


N=64
#N=30
#phi=0.69#number_density
#phi=2.000
phi=0.25
L=(N/phi)**(1/3)

for name in range(0,200):
    #L=L-0.1
    print (L)
    newL=round(L,3)
    print (newL)
    
    for config in range(1,99):
        
        data=pylab.loadtxt('traj_MC_ML_'+str(newL)+'_'+str(config)+'.dat')
        x=data[:,0]
        y=data[:,1]
        z=data[:,2]
        alpha=data[:,3]
        beta=data[:,4]
        gamma=data[:,5]
        extended_x=[]
        extended_y=[]
        extended_z=[]

        for i in range(0,len(x)):
            x_object_n, y_object_n,z_object_n=rotate_3d(x_object,y_object,z_object,alpha[i],beta[i],gamma[i])
            x_object_n, y_object_n, z_object_n=translate_3d(x_object_n, y_object_n,z_object_n,x[i],y[i],z[i])
            for j in range(0,len(x_object_n)):
                extended_x.append(x_object_n[j])
                extended_y.append(y_object_n[j])
                extended_z.append(z_object_n[j])

        s=64*9 
        g.write(str(s)+'\n')
        g.write('xyz'+'\n')
        for i in range(0,len(extended_x)):
            g.write('C   '+str(extended_x[i])+'   '+str(extended_y[i])+'   '+str(extended_z[i])+'\n')
    
    #L=L-((4)/(3*L*L))
    L=(64/((64/(L**3))+0.075))**(1/3)
    #print (round(L,2))
    #L=L-0.1
g.close()
#plt.scatter(extended_x,extended_y, color='red', alpha=0.1)

#plt.show()




