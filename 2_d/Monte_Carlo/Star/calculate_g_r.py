from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pylab




def cal_grtheta(M,N,NP,L,binr,bintheta,filename):

    r=np.linspace(0.1,5,binr)
    
    angle=np.linspace(0,360,bintheta)
    
    dis=np.zeros((binr,bintheta))
    for i in range(M,N):
        print (i)
        
        data=pylab.loadtxt('traj_MC_'+str(filename)+'_'+str(L)+'_'+str(i)+'.dat')
        x=data[:,0]
        y=data[:,1]
        theta=data[:,2]
        for j in range(0,NP):
            for k in range(j,NP):
                dx=abs(x[j]-x[k])
                dy=abs(y[j]-y[k])
                if dx>L/2:
                    dx=L-dx
                if dy>L/2:
                    dy=L-dy
                d=((dx)**2+(dy)**2)**0.5
                dtheta=theta[j]-theta[k]
                if dtheta<0:
                    dtheta=2*np.pi-abs(dtheta)
                dtheta=dtheta*(180/np.pi)


            #d=((x[j]-x[k])**2+(y[j]-y[k])**2)**0.5
               
                for l in range(0,len(r)-1):
                    for the in range(0,len(angle)-1):
                        if d>r[l] and d<r[l+1]:
                            if dtheta>angle[the] and dtheta<angle[the+1]:
                                dis[l][the]=dis[l][the]+1


    for i in range(0,binr):
        for j in range(0,bintheta):
            dis[i][j]=dis[i][j]/((N-M)*2*3.14*r[i])

    return r, angle, dis
 
r, angle, dis = cal_grtheta(1,100,64,9.92,50,10,'ML')

for i in range(0,9):
    plt.plot(r, dis[:,i],color='black',label='ML')



r, angle, dis = cal_grtheta(1,100,64,9.92,50,10,'NORMAL')

for i in range(0,9):
    plt.plot(r, dis[:,i],color='red',label='NORMAL')

plt.legend()
plt.show()







