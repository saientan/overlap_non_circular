from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pylab




def calc_g_r(name):
    M=1     
    N=100          
    NP=64
    L=10.79 
    r=np.linspace(0.001,5,30)
    dis=np.zeros(len(r))
    for i in range(M,N):
        print (i)
        data=pylab.loadtxt('traj_MC_'+name+'_'+str(L)+'_'+str(i)+'.dat')
        x=data[:,0]
        y=data[:,1]
        for j in range(0,NP):
            for k in range(j,NP):
                dx=abs(x[j]-x[k])
                dy=abs(y[j]-y[k])
                if dx>L/2:
                    dx=L-dx
                if dy>L/2:
                    dy=L-dy
                d=((dx)**2+(dy)**2)**0.5

            #d=((x[j]-x[k])**2+(y[j]-y[k])**2)**0.5
                for l in range(0,len(r)-1):
                    if d>r[l] and d<r[l+1]:
                        dis[l+1]=dis[l+1]+1


    for i in range(0,len(dis)):
        dis[i]=dis[i]/((N-M)*2*3.14*r[i])
        dis[i]=dis[i]/((64/2)*(r[2]-r[1])*(64/(L*L)))

    #plt.plot(r,dis)
    return r,dis


plt.figure(figsize=(7, 6))

r,dis=calc_g_r('NORMAL')
plt.plot(r,dis, '-o',color='red', label='Explicit Distance Calculation')

r,dis=calc_g_r('ML_Decision_Tree')
plt.plot(r,dis, '-o', color='blue',label='Decision Tree')
#plt.legend(loc=4, fontsize=15)


r,dis=calc_g_r('ML_QDA')
plt.plot(r,dis, '-o', color='green',label='QDA')
#plt.legend(loc=4, fontsize=15)


r,dis=calc_g_r('ML_Naive_Bayes')
plt.plot(r,dis, '-o', color='violet',label='Naive Bayes')
#plt.legend(loc=4, fontsize=15)


r,dis=calc_g_r('ML_Grad_Boost')
plt.plot(r,dis, '-o', color='black',label='Grad Boost')
plt.legend(loc=4, fontsize=15)




plt.xlabel(r"$r$",fontsize=15)
plt.ylabel(r"$g(r)$",fontsize=15)
plt.tick_params(direction='in', length=6, width=2, colors='black', labelsize=15, grid_color='black')
#axes[1].legend(fontsize=15, loc=2)

plt.tight_layout()
plt.savefig('Figure_g_r_circle.png', dpi=200)
plt.show()
