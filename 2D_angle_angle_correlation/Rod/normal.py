from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random
import pylab
import time
import os
import sklearn
import sklearn.linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split

####################Translation##############
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
    dx = x - xnew
    dy = y - ynew

    if dx > L/2:
        dx = L - dx
        xnew = x - dx
    if dy > L/2:
        dy = L - dy
        ynew = y - dy

    if dx < -L/2:
        dx = -L - dx
        xnew = x - dx
    if dy < -L/2:
        dy = -L - dy
        ynew = y - dy

    ov = 0
    
    #diff_theta=theta-thetanew
    #if diff_theta<0:
    #    diff_theta=(2*np.pi)-abs(diff_theta)

    x_object_new_1, y_object_new_1 = rotate(x_object, y_object, theta)
    x_object_new_1, y_object_new_1 = translate(x_object_new_1, y_object_new_1, x, y)

    x_object_new_2, y_object_new_2 = rotate(x_object, y_object, thetanew)
    x_object_new_2, y_object_new_2 = translate(x_object_new_2, y_object_new_2, xnew, ynew)

    for i in range(0,len(x_object_new_1)):
        for j in range(0,len(x_object_new_2)):
        
            d = ((x_object_new_1[i]-x_object_new_2[j])**2 + (y_object_new_1[i]-y_object_new_2[j])**2)**0.5
            if d <= 0.11111111111111116: #######Which Cutoff Between the Particles########
                ov = 1
                break
        if ov == 1:
            break
    return ov

######################Slow_Compression_of_Box###################
def compress(L,dcomp,x,y,theta):
    newL = L - dcomp
    for i in range(0,len(x)):
        x[i] =  (x[i]/L)*newL
        y[i] = (y[i]/L)*newL
        
    return x, y, theta, newL

#############MONTE_CARLO_MOVE##############
def monte_carlo_move(MC,dx,dy,dtheta,L,x,y,theta,freq,filename):
    acc=0

    for i in range(1,MC):
        number = random.randint(0,N-1)
        xnew = x[number] + random.uniform(-dx, dx)
        ynew = y[number] + random.uniform(-dy, dy)
        thetanew = theta[number] + random.uniform(-dtheta, dtheta)
        
        if xnew > L:
            xnew = xnew-L            
        if xnew < 0:
            xnew = L-abs(xnew)
            
        if ynew > L:
            ynew = ynew-L
        if ynew < 0:
            ynew = L-abs(ynew)

        if thetanew > 2*np.pi:
            thetanew = thetanew - (2*np.pi)
        if thetanew < 0:
            thetanew = (2*np.pi) + thetanew
   
        check = 0
        #?
        for j in range(0,N):  # for every particle other than the selected one
            if number != j:
                o_lap = overlap(xnew, ynew, thetanew, x[j], y[j], theta[j], L)
                if o_lap == 1:
                    check = 1
                    break
                
        if check == 0:
            x[number] = xnew
            y[number] = ynew
            theta[number] = thetanew
            acc = acc + 1
            
        # Collecting generated dx, dy and dtheta for all N particles after every freq steps
        if i%freq == 0:  
            s = './' + str(run) + '_' + filename + '_' + str(round(L,2)) + '/traj_MC_' + str(filename) + '_' + str(round(L,2)) + '_' + str(int(i/freq)) + '.dat'
            f = open(s,'w')
            
            for k in range(0,N):
                f.write(str(x[k]) + '   ' + str(y[k]) + '   ' + str(theta[k]) + '\n')
        
        if i%1000 == 0:
            print(i, acc/i)
        
    return x, y, theta, L

# Area_fraction = [0.2]

for run in range(1,10):
    Area_frac = 0.20
    D = 0.1111111111111
    num_disk = 9
    phi = 4*Area_frac/(num_disk*np.pi*D*D)

    N = 64
    L = (N/phi)**0.5
    print(L)

    data_object = pylab.loadtxt('./coord.dat')
    x_object = data_object[:,0]
    y_object = data_object[:,1]

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
        newx = random.uniform(0,L)
        newy = random.uniform(0,L)
        newtheta = random.uniform(0,2*np.pi)
        check = 0 

        for j in range(0,len(x)):
            o_lap = overlap(newx,newy,newtheta,x[j],y[j],theta[j],L)

            if o_lap == 1:            
                check = 1
                break
        
        if check == 0: # append only when overlap is 0
            x.append(newx)
            y.append(newy)
            theta.append(newtheta)
            count = count + 1

        print(count, end = " ")

        if count == N-1:
            break

    initial_snap = open('./initial_snap.dat','w')

    extended_x = []
    extended_y = []

    for i in range(0,len(x)):
        x_object_n, y_object_n = rotate(x_object, y_object, theta[i])
        x_object_n, y_object_n = translate(x_object_n, y_object_n, x[i], y[i])

        for j in range(0,len(x_object_n)):
            extended_x.append(x_object_n[j])
            extended_y.append(y_object_n[j])

    plt.scatter(extended_x, extended_y, color = 'red', alpha=0.1)
    
    for i in range(0,len(extended_x)):
        initial_snap.write(str(extended_x[i])+'   '+str(extended_y[i])+'\n')
    initial_snap.close()

    start_ML = time.time()

    for i in range(0,1):
        os.makedirs(str(run) + '_NORMAL_' + str(round(L,2)))
        x, y, theta, L = monte_carlo_move(100000, 0.4, 0.4, 0.17, L, x, y, theta, 1000, 'NORMAL') # MC, dx, dy, dtheta, L, x, y, theta, freq, filename
        # x, y, theta, L = compress(L, 0.1, x, y, theta) # L, dcomp, x, y, theta

    end_ML = time.time()
    print(end_ML - start_ML)