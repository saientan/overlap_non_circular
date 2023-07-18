from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random
import pylab

N=64    
#N=30
#phi=0.69#number_density
phi=0.69
L=(N/phi)**0.5
print (L)

#########################The_Extended_Object##################################

data_object=pylab.loadtxt('coord_star.dat')
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

def overlap(x,y,theta,xnew,ynew,thetanew):
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
            if d<0.095: #######Which Cutoff Between the Particles########
                overlap=1 
                break
    return overlap

###################Machine_Learning_Overlap_Detection##########################

def overlap_ML(x,y,theta,xnew,ynew,thetanew,model):

    dx=abs(x-xnew)
    dy=abs(y-ynew)
    if dx>L/2:
        dx=L-dx
    if dy>L/2:
        dy=L-dy
 
    diff_theta=theta-thetanew
    if diff_theta<0:
        diff_theta=2*np.pi-abs(diff_theta)
    ml_in=[[dx, dy, diff_theta]]
    #ml_in=ml_in.reshape(1, -1)
    ov_ml=model.predict(ml_in)
    if ov_ml<0.5:
        ov_ml=0
    if ov_ml>=0.5:
        ov_ml=1
    return ov_ml

'''
data=pylab.loadtxt('training_data.dat')
xml=data[:,0:3]
yml=data[:,3]
x_train, x_test, y_train, y_test = train_test_split(xml, yml, test_size=0.4 ,random_state=0)
model =  RandomForestClassifier(class_weight="balanced")
model.fit(x_train,y_train.ravel())
'''
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
        o_lap=overlap(newx,newy,newtheta,x[j],y[j],theta[j])
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

plt.scatter(extended_x,extended_y)
plt.show()

#############MONTE_CARLO_MOVE##############
MC=100000
dx=0.4 
dy=0.4        
dtheta=0.17
acc=0

#f=open('traj.dat','w')
for i in range(1,MC):
    number=random.randint(0,N-1)
    xnew=x[number]+random.uniform(-dx,dx)
    ynew=y[number]+random.uniform(-dy,dy)
    thetanew=theta[number]+random.uniform(-dtheta,dtheta)
    if xnew>L:
        xnew=xnew-L
    if xnew<0:
        xnew=L-abs(xnew)
    if ynew>L:
        ynew=ynew-L
    if ynew<0:
        ynew=L-abs(ynew)
   
    check=0
    for j in range(0,N):

        if number!=j:
            o_lap=overlap(xnew,ynew,thetanew,x[j],y[j],theta[j])
            if o_lap==1:
                check=1
    if check==0:
        x[number]=xnew
        y[number]=ynew
        theta[number]=thetanew
        acc=acc+1
    if i%1000==0:
        s='traj_'+str(int(i/1000))+'.dat'
        f=open(s,'w')
        for k in range(0,N):
            f.write(str(x[k])+'   '+str(y[k])+'   '+str(theta[k])+'\n')
    print (i, acc/i)
        








