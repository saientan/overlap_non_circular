from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pylab




def cal_theta(M,N,NP,L,bintheta,name):

    #r=np.linspace(0.1,5,binr)
    
    angle=np.linspace(0,360,bintheta)
    #angle=np.linspace(-1,1,bintheta) 
    dis=np.zeros(bintheta)
    for i in range(M,N):
        print (i)
        
        data=pylab.loadtxt('traj_MC_'+str(name)+'_'+str(L)+'_'+str(i)+'.dat')
        theta=data[:,2]*(180/np.pi)
        
        for j in range(0,NP):                         
            for the in range(0,len(angle)-1):
                if theta[j]>angle[the] and theta[j]<angle[the+1]:
                                dis[the]=dis[the]+1


    return angle, dis



 
angle, dis = cal_theta(1,100,64,9.63,100,'NORMAL')

a=[]
b=[]

for i in range(0,len(dis)-1):
    a.append(angle[i])
    b.append(dis[i])

plt.plot(a,b/sum(b))
#plt.ylim(0,0.1)


angle, dis = cal_theta(1,100,64,9.63,100,'ML')

a=[]
b=[]

for i in range(0,len(dis)-1):
    a.append(angle[i])
    b.append(dis[i])

plt.plot(a,b/sum(b), color='red')
#plt.ylim(0,0.1)

plt.show()







