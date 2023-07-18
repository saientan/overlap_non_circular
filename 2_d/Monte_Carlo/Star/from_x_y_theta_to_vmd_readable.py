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
import matplotlib.pyplot as plt
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

data_object=pylab.loadtxt('coord.dat')
x_object=data_object[:,0]
y_object=data_object[:,1]

#full_object=pylab.loadtxt('full_star_coord.dat')
#x_full_object=full_object[:,0]
#y_full_object=full_object[:,1]

def convert(name):

    g=open('for_vmd_ML_'+str(name)+'.xyz','w')
    L=10.79

    for abc in range(0,1):
    #L=L-0.1
        print (L)
        L=round(L,2)
        print (L)
        for config in range(1,100):
        
            data=pylab.loadtxt('traj_MC_ML_'+str(name)+'_'+str(L)+'_'+str(config)+'.dat')
            x=data[:,0]
            y=data[:,1]
            theta=data[:,2]
            extended_x=[]
            extended_y=[]

            for i in range(0,len(x)):
                x_object_n, y_object_n=rotate(x_object,y_object,theta[i])
                x_object_n, y_object_n=translate(x_object_n, y_object_n,x[i],y[i])
                for j in range(0,len(x_object_n)):
                    extended_x.append(x_object_n[j])
                    extended_y.append(y_object_n[j])

            s=64*31
            g.write(str(s)+'\n')
            g.write('xyz'+'\n')
            for i in range(0,len(extended_x)):
                g.write('C   '+str(extended_x[i])+'   '+str(extended_y[i])+'   0.000'+'\n')


    g.close()

convert("Grad_Boost")
convert("Decision_Tree")
convert("Naive_Bayes")
convert("QDA")

#plt.scatter(extended_x,extended_y, color='red', alpha=0.1)

#plt.show()




