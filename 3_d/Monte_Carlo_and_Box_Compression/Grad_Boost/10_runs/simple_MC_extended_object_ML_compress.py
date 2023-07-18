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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn.ensemble import HistGradientBoostingClassifier
import time
#########################################################################
N=64    
#N=30
#phi=0.69#number_density
phi=0.25    
L=(N/phi)**(1/3)
print (L)

#########################The_Extended_Object##################################

data_object=pylab.loadtxt('coord.dat')
x_object=data_object[:,0]
y_object=data_object[:,1]
z_object=data_object[:,2]

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

####################Extended_Object_Overlap_Detection##################

def overlap(x,y,z,alpha,beta,gamma,xnew,ynew,znew,alphanew,betanew,gammanew,L):
    dx=abs(x-xnew)
    dy=abs(y-ynew)
    dz=abs(z-znew)
    if dx>L/2:
        dx=L-dx
    if dy>L/2:
        dy=L-dy
    if dz>L/2:
        dz=L-dz
    overlap=0
    diff_alpha=alpha-alphanew
    diff_beta=beta-betanew
    diff_gamma=gamma-gammanew
    #if diff_theta<0:
    #    diff_theta=2*np.pi-abs(diff_theta)


    x_object_new, y_object_new,z_object_new=rotate_3d(x_object,y_object,z_object,diff_alpha,diff_beta,diff_gamma)
    x_object_new, y_object_new,z_object_new=translate_3d(x_object_new, y_object_new,z_object_new,dx,dy,dz)


    for i in range(0,len(x_object)):
        for j in range(0,len(x_object_new)):
            d=((x_object[i]-x_object_new[j])**2+(y_object[i]-y_object_new[j])**2+(z_object[i]-z_object_new[j])**2)**0.5
            if d<=0.11111111111111116: #######Which Cutoff Between the Particles########
                overlap=1
                break
    return overlap

###################Machine_Learning_Overlap_Detection##########################

def overlap_ML(x,y,z,alpha,beta,gamma,xnew,ynew,znew,alphanew,betanew,gammanew,model,L):

    #dx=abs(x-xnew)
    dx=x-xnew
    #dy=abs(y-ynew)
    dy=y-ynew

    dz=z-znew

    if dx>L/2:
        dx=L-dx
    if dy>L/2:
        dy=L-dy
    if dz>L/2:
        dz=L-dz

    if dx<-L/2:
        dx=-L-dx
    if dy<-L/2:
        dy=-L-dy
    if dz<-L/2:
        dz=-L-dz


 
    diff_alpha=alpha-alphanew
    diff_beta=beta-betanew
    diff_gamma=gamma-gammanew
    #if diff_theta<0:
    #    diff_theta=2*np.pi-abs(diff_theta)
    ml_in=[[dx, dy,dz, diff_alpha, diff_beta, diff_gamma]]
    #ml_in=ml_in.reshape(1, -1)
    ov_ml=model.predict(ml_in)
    #if ov_ml<0.5:
    #    ov_ml=0
    #if ov_ml>=0.5:
    #    ov_ml=1
    #print (ov_ml)
    return ov_ml


#data_ML=pylab.loadtxt('training_data.dat')
data_ML=pylab.loadtxt('new_training_data.dat')
#x_train=data_ML[:,0:3]
#y_train=data_ML[:,3]

xml=data_ML[:,0:6]
yml=data_ML[:,6]
x_train, x_test, y_train, y_test = train_test_split(xml, yml, test_size=0.1 ,random_state=0)
#model =  RandomForestClassifier(class_weight="balanced")
#model =  RandomForestClassifier()

model = GradientBoostingClassifier()

#model=KNeighborsClassifier(3)
#model = QuadraticDiscriminantAnalysis()

#model=HistGradientBoostingClassifier()
model.fit(x_train,y_train.ravel())
pred_train=model.predict(x_train)
accuracy=0
for i in range(0,len(x_train)):
    if y_train[i]==pred_train[i]:
        accuracy=accuracy+1
print ("train_accuracy", accuracy/len(x_train))
pred_test=model.predict(x_test)
accuracy=0
for i in range(0,len(x_test)):
    if y_test[i]==pred_test[i]:
        accuracy=accuracy+1
print ("test_accuracy", accuracy/len(x_test))

######################Slow_Compression_of_Box###################
def compress(L,dcomp,x,y,z,alpha,beta,gamma):
    newL=(64/((64/(L**3))+dcomp))**(1/3)
    #newL=L-((dcomp)/(3*L*L))
    #newL=L-dcomp
    for i in range(0,len(x)):
        x[i]=(x[i]/L)*newL
        y[i]=(y[i]/L)*newL
        z[i]=(z[i]/L)*newL
    return x, y,z,alpha,beta,gamma, newL

def relax(L,dcomp,x,y,z,alpha,beta,gamma):
    newL=L+dcomp
    for i in range(0,len(x)):
        x[i]=(x[i]/L)*newL
        y[i]=(y[i]/L)*newL
        z[i]=(z[i]/L)*newL
    return x, y,z,alpha,beta,gamma, newL


#############Random_initial_position_of_extended_object##############

M=100000000
count=0

x=[]
y=[]
z=[]
alpha=[]
beta=[]
gamma=[]
x.append(random.uniform(0,L))
y.append(random.uniform(0,L))
z.append(random.uniform(0,L))

alpha.append(np.random.uniform(0,2*np.pi))
beta.append(np.random.uniform(0,np.pi))
gamma.append(np.random.uniform(0,2*np.pi))


for i in range(0,M):
    newx=random.uniform(0,L)
    newy=random.uniform(0,L)
    newz=random.uniform(0,L)
    newalpha=random.uniform(0,2*np.pi)
    newbeta=random.uniform(0,np.pi)
    newgamma=random.uniform(0,2*np.pi)
    check=0
    for j in range(0,len(x)):
        #o_lap=overlap(newx,newy,newtheta,x[j],y[j],theta[j])
        o_lap=overlap(newx,newy,newz,newalpha,newbeta,newgamma,x[j],y[j],z[j],alpha[j],beta[j],gamma[j],L)
        #o_lap=overlap_ML(newx,newy,newz,newalpha,newbeta,newgamma,x[j],y[j],z[j],alpha[j],beta[j],gamma[j],model,L)
        #print ("o_la",o_lap)
        if o_lap==1:
            
            check=1
            
    if check==0:

        x.append(newx)
        y.append(newy)
        z.append(newz)
        alpha.append(newalpha)
        beta.append(newbeta)
        gamma.append(newgamma)
        count=count+1
    print (count)
    if count==N-1:
        break
        
#print (len(x))
#plt.scatter(x,y)


extended_x=[]
extended_y=[]
extended_z=[]

for i in range(0,len(x)):
    x_object_n, y_object_n, z_object_n=rotate_3d(x_object,y_object,z_object,alpha[i],beta[i],gamma[i])
    x_object_n, y_object_n,z_object_n=translate_3d(x_object_n,y_object_n,z_object_n,x[i],y[i],z[i])
    for j in range(0,len(x_object_n)):
        extended_x.append(x_object_n[j])
        extended_y.append(y_object_n[j])
        extended_z.append(z_object_n[j])
plt.scatter(extended_x,extended_y,extended_z, color='red', alpha=0.1)

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

def monte_carlo_move(MC,dx,dy,dz,dalpha,dbeta,dgamma,L,x,y,z,alpha,beta,gamma):


    acc=0

#f=open('traj.dat','w')
    for i in range(1,MC):
        number=random.randint(0,N-1)
        xnew=x[number]+random.uniform(-dx,dx)
        ynew=y[number]+random.uniform(-dy,dy)
        znew=z[number]+random.uniform(-dz,dz)

        alphanew=alpha[number]+random.uniform(-dalpha,dalpha)
        betanew=beta[number]+random.uniform(-dbeta,dbeta)
        gammanew=alpha[number]+random.uniform(-dgamma,dgamma)
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
        if znew>L:
            znew=znew-L
        if znew<0:
            znew=L-abs(znew)

   
        if alphanew>2*np.pi:
            alphanew=alphanew-(2*np.pi)
        if alphanew<0:
            alphanew=(2*np.pi)+alphanew

        if betanew>np.pi:
            betanew=betanew-(np.pi)
        if betanew<0:
            betanew=(np.pi)+betanew

        if gammanew>2*np.pi:
            gammanew=gammanew-(2*np.pi)
        if gammanew<0:
            gammanew=(2*np.pi)+gammanew



        check=0
        for j in range(0,N):

            if number!=j:
            #o_lap=overlap(xnew,ynew,thetanew,x[j],y[j],theta[j])
                o_lap=overlap_ML(xnew,ynew,znew,alphanew,betanew,gammanew,x[j],y[j],z[j],alpha[j],beta[j],gamma[j],model,L)
                if o_lap==1:
                    check=1
        if check==0:
            x[number]=xnew
            y[number]=ynew
            z[number]=znew
            alpha[number]=alphanew
            beta[number]=betanew
            gamma[number]=gammanew
            acc=acc+1
        if i%1000==0:
            s='traj_MC_ML_'+str(round(L,3))+'_'+str(int(i/1000))+'.dat'
            f=open(s,'w')
            for k in range(0,N):
                f.write(str(x[k])+'   '+str(y[k])+'   '+str(z[k])+'   '+str(alpha[k])+'   '+str(beta[k])+'   '+str(gamma[k])+'\n')
            #f.write('C   '+str(x[k])+'   '+str(y[k])+'   '+'0.000'+'\n')
        print (i, acc/i,L,dx,dalpha) 
    acc=acc/MC
    return x,y,z,alpha,beta,gamma,L,acc        
    

#start_ML = time.time()

for i in range(0,200): #(0,30)
    mct=0.5
    mcr=1.21
    for j in range(0,30):
        x,y,z,alpha,beta,gamma, L,acc=monte_carlo_move(200,mct,mct,mct,mcr,mcr,mcr,L,x,y,z,alpha,beta,gamma)
        if acc>0.5:
            break
        else:
            if mct>0.2:
                mct=mct-0.1
            if mct<0.2 and mct > 0.05:
                mct=mct-0.01
            if mcr>0.2:
                mcr=mcr-0.1
            if mcr<0.2 and mcr > 0.05:
                mcr=mcr-0.01
    start_ML = time.time()
    x,y,z,alpha,beta,gamma, L,acc=monte_carlo_move(100000,mct,mct,mct,mcr,mcr,mcr,L,x,y,z,alpha,beta,gamma)
    end_ML = time.time()
    x, y,z,alpha,beta,gamma, L= compress(L,0.075,x,y,z,alpha,beta,gamma)

#end_ML = time.time()

time_taken= (end_ML-start_ML)
print ("time taken", time_taken)




