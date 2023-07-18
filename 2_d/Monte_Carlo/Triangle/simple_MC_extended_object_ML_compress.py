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
import time
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

##########################################################

names = [
#    "Nearest_Neighbors",
#    "Linear_SVM",
#    "RBF_SVM",
    "Decision_Tree",
#    "Random_Forest",
#    "Neural_Net",
#    "AdaBoost",
    "Naive_Bayes",
    "QDA",
    "Grad_Boost",
#    "HistGradientBoostingClassifier",
]

classifiers = [
#    KNeighborsClassifier(3),
#    SVC(kernel="linear", C=0.025),
#    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
#    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#    MLPClassifier(alpha=1, max_iter=1000),
#    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    GradientBoostingClassifier(),
#    HistGradientBoostingClassifier()
]


def RUN_ALL(model,model_name):

#########################################################################
    N=64    
    #N=30
    #phi=0.69#number_density
    phi=0.55    
    L=(N/phi)**0.5
    print (L)

#########################The_Extended_Object##################################

    data_object=pylab.loadtxt('coord.dat')
    x_object=data_object[:,0]
    y_object=data_object[:,1]


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
        dx=x-xnew
        dy=y-ynew
        if dx>L/2:
            dx=L-dx
            xnew=x-dx
        if dy>L/2:
            dy=L-dy
            ynew=y-dy
        if dx<-L/2:
            dx=-L-dx
            xnew=x-dx
        if dy<-L/2:
            dy=-L-dy
            ynew=y-dy


    #if dy>L/2:
    #    dy=L-dy
        ov=0
    #diff_theta=theta-thetanew
    #if diff_theta<0:
    #    diff_theta=(2*np.pi)-abs(diff_theta)


        x_object_new_1, y_object_new_1=rotate(x_object,y_object,theta)
        x_object_new_1, y_object_new_1=translate(x_object_new_1, y_object_new_1,x,y)

        x_object_new_2, y_object_new_2=rotate(x_object,y_object,thetanew)
        x_object_new_2, y_object_new_2=translate(x_object_new_2, y_object_new_2,xnew,ynew)




        for i in range(0,len(x_object_new_1)):
            for j in range(0,len(x_object_new_2)):
                d=((x_object_new_1[i]-x_object_new_2[j])**2+(y_object_new_1[i]-y_object_new_2[j])**2)**0.5
            #print (d)
                if d<=0.11111111111111116: #######Which Cutoff Between the Particles########
                    ov=1
                    break
        return ov


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


#model =  RandomForestClassifier(class_weight="balanced")
#model =  RandomForestClassifier()

#model = GradientBoostingClassifier()

#model.fit(x_train,y_train.ravel())
#pred_train=model.predict(x_train)
#accuracy=0
#for i in range(0,len(x_train)):
#    if y_train[i]==pred_train[i]:
#        accuracy=accuracy+1
#print ("train_accuracy", accuracy/len(x_train))
#pred_test=model.predict(x_test)
#accuracy=0
#for i in range(0,len(x_test)):
#    if y_test[i]==pred_test[i]:
#        accuracy=accuracy+1
#print ("test_accuracy", accuracy/len(x_test))

######################Slow_Compression_of_Box###################
    def compress(L,dcomp,x,y,theta):
        newL=L-dcomp
        for i in range(0,len(x)):
            x[i]=(x[i]/L)*newL
            y[i]=(y[i]/L)*newL
        return x, y, theta, newL



#############Random_initial_position_of_extended_object##############

    M=100000000
    count=0

    x=[]
    y=[]
    theta=[]
    x.append(random.uniform(0,L))
    y.append(random.uniform(0,L))
    theta.append(random.uniform(0,2*np.pi))
    for i in range(0,M):
        newx=random.uniform(0,L)
        newy=random.uniform(0,L)
        newtheta=random.uniform(0,2*np.pi)
        check=0
        for j in range(0,len(x)):
            o_lap=overlap(newx,newy,newtheta,x[j],y[j],theta[j],L)
        #o_lap=overlap_ML(newx,newy,newtheta,x[j],y[j],theta[j],model,L)
        #print ("o_la",o_lap)
            if o_lap==1:
            
                check=1
            
        if check==0:

            x.append(newx)
            y.append(newy)
            theta.append(newtheta)
            count=count+1
        print (count)
        if count==N-1:
            break
        
#print (len(x))
#plt.scatter(x,y)


    extended_x=[]
    extended_y=[]

    for i in range(0,len(x)):
        x_object_n, y_object_n=rotate(x_object,y_object,theta[i])
        x_object_n, y_object_n=translate(x_object_n, y_object_n,x[i],y[i])
        for j in range(0,len(x_object_n)):
            extended_x.append(x_object_n[j])
            extended_y.append(y_object_n[j])
    plt.scatter(extended_x,extended_y, color='red', alpha=0.1)

    '''
    extended_x=[]
    extended_y=[]

    for i in range(0,len(x)):
        x_object_n, y_object_n=rotate(x_full_object,y_full_object,theta[i])
        x_object_n, y_object_n=translate(x_object_n, y_object_n,x[i],y[i])
        for j in range(0,len(x_object_n)):
            extended_x.append(x_object_n[j])
            extended_y.append(y_object_n[j])
    plt.scatter(extended_x,extended_y,color='red', alpha=0.1)

    '''

    #plt.show()

    #############MONTE_CARLO_MOVE##############

    def monte_carlo_move(MC,dx,dy,dtheta,L,x,y,thetai, ML_Model,model_name):


        acc=0

#f=open('traj.dat','w')
        for i in range(1,MC):
            number=random.randint(0,N-1)
            xnew=x[number]+random.uniform(-dx,dx)
            ynew=y[number]+random.uniform(-dy,dy)
            thetanew=theta[number]+random.uniform(-dtheta,dtheta)
        #if xnew>L or xnew <0:
        #    xnew=x[number]

        #if ynew>L or ynew <0:
        #    ynew=y[number]
            if xnew>L:
                xnew=xnew-L            
            if xnew<0:
                xnew=L-abs(xnew)
            if ynew>L:
                ynew=ynew-L
            if ynew<0:
                ynew=L-abs(ynew)
   
            if thetanew>2*np.pi:
                thetanew=thetanew-(2*np.pi)
            if thetanew<0:
                thetanew=(2*np.pi)+thetanew
            check=0
            for j in range(0,N):

                if number!=j:
            #o_lap=overlap(xnew,ynew,thetanew,x[j],y[j],theta[j])
                   o_lap=overlap_ML(xnew,ynew,thetanew,x[j],y[j],theta[j],ML_Model,L)
                   if o_lap==1:
                       check=1
            if check==0:
                x[number]=xnew
                y[number]=ynew
                theta[number]=thetanew
                acc=acc+1
            if i%1000==0:
                s='traj_MC_ML_'+str(model_name)+'_'+str(round(L,2))+'_'+str(int(i/1000))+'.dat'
                f=open(s,'w')
                for k in range(0,N):
                    f.write(str(x[k])+'   '+str(y[k])+'   '+str(theta[k])+'\n')
            #f.write('C   '+str(x[k])+'   '+str(y[k])+'   '+'0.000'+'\n')
            print (i, acc/i)
        
        return x,y,theta,L

    start_ML = time.time()



    
    for i in range(0,1):
        x,y,theta,L=monte_carlo_move(100000,0.4,0.4,0.17,L,x,y,theta,model,model_name)
        x, y, theta, L= compress(L,0.1,x,y,theta)
    
    end_ML = time.time()
    time_taken= (end_ML-start_ML)
    abcd=open("time_taken_"+str(model_name),'w')
    abcd.write(str(time_taken))

data_ML=pylab.loadtxt('new_training_data.dat')
#x_train=data_ML[:,0:3]
#y_train=data_ML[:,3]

xml=data_ML[:,0:3]
yml=data_ML[:,3]
x_train, x_test, y_train, y_test = train_test_split(xml, yml, test_size=0.1 ,random_state=0)


for i in range(0,len(names)):
    model=classifiers[i]
    model.fit(x_train,y_train.ravel())
    model_name=names[i]
    RUN_ALL(model,model_name)





