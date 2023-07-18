import numpy as np
import matplotlib.pyplot as plt
import pylab

names = [
    "Nearest Neighbors",
#    "Neural Net",
    "RBF SVM",
#    "Linear SVM",
    "AdaBoost",
    "Random Forest",
#    "HistGradientBoostingClassifier",
    "Decision Tree",
    "QDA",
    "Grad Boost",
    "Naive Bayes",
]
print (len(names))

M=10

def calculate(name,N):

    g=open('temporary.dat','w')
    for i in range(1,M):
        f=open('time_taken_'+str(i)+'.dat')
        l_f=f.readlines()
        for j in range(0,len(l_f)):
            if name in l_f[j]:
                g.write(l_f[j])
        f.close()         
    g.close()
    
    
    x=[]
    y=[]
    g=open('temporary.dat','r')
    l_g=g.readlines()
    l=len(l_g)

    for i in range(0,l):
        if float(l_g[i].split()[-3])==N:
            x.append(float(l_g[i].split()[-1]))
            y.append(float(l_g[i].split()[-2]))

    g.close()
    

    
    #print (x)
    #print(y)
    
    for i in range(0,len(x)):
        x[i]=x[i]*1000
        y[i]=y[i]*1000
    avgx=np.average(x)
    avgy=np.average(y)
    stdx=np.std(x)
    stdy=np.std(y)
    print (name,N, stdx,stdy)
    return avgx,stdx,avgy,stdy
    

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))


big=[25, 49 ,81, 121, 169, 225, 289 ,361, 441, 529, 625, 729 ,841]
cig=np.zeros(len(big))
for i in range(0,len(big)):
    cig[i]=big[i]**0.5
#print(big)
for k in range(0,2):
    for i in range(k*4,(k+1)*4):
    #for i in range(0,len(names)):
        avgx=[]
        stdx=[]
        avgy=[]
        stdy=[]
        for j in range(0,len(big)):
            #print(i)
            a,b,c,d=calculate(names[i],big[j]) 
            avgx.append(a)
            stdx.append(b)
            avgy.append(c)
            stdy.append(d)
        axes[k].errorbar(cig,avgx,yerr=stdx, fmt='o-', markersize=8, capsize=10, label=names[i])

    axes[k].errorbar(cig,avgy,yerr=stdy, fmt='o-', markersize=8, capsize=10, label='Explicit distance calculation')
    #axes[k].set_xlabel("Number of disk used to represent the rigid body",fontsize=15)
    axes[k].set_ylabel("Single overlap calculation time (ms)",fontsize=15)
    axes[k].tick_params(direction='in', length=6, width=2, colors='black', labelsize=15, grid_color='black')
    axes[k].legend(fontsize=15)
    #axes[k].set_xlim(5,800)
fig.text(0.5,0.02,"Number of disks used to represent the rigid body",fontsize=15,ha='center')
fig.text(0.02,0.96,"(a)",fontsize=15)
fig.text(0.52,0.96,"(b)",fontsize=15)
fig.tight_layout()
plt.show()





















