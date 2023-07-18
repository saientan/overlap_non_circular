import numpy as np
import matplotlib.pyplot as plt
import pylab
#N=149
#M=10

def plot_all(location,name):

    density=pylab.loadtxt(str(location)+'/1/density_vs_nematic_order_parameters.dat')
    density=density[:,0]
    density=density*9*(4/3)*3.14*((0.11/2)**3)
    density=density/0.1309704515597676
    
    N=len(density)
    avg_order=[]
    std_order=[]
    for i in range(0,N):
        order=[]
        for j in range(1,11):
            data=pylab.loadtxt(str(location)+'/'+str(j)+'/density_vs_nematic_order_parameters.dat')
            order.append(data[:,1][i])
        avg=np.average(order)
        std=np.std(order)
        avg_order.append(avg)
        std_order.append(std)

   # plt.errorbar(density,avg_order,yerr=std_order, fmt='o-', markersize=8, capsize=10, label=str(name))
    a=[]
    b=[]
    c=[]
    for i in range(0,len(density)-40,5):
        a.append(density[i])
        b.append(avg_order[i])
        c.append(std_order[i])
    return (a,b,c)
    #plt.errorbar(a,b,yerr=c,  label=str(name))
   #plt.errorbar(density,avg_order,yerr=std_order,  label=str(name))


a,b,c=plot_all('Normal/10_runs','Explcit Distance Calculation')
plt.errorbar(a,b, yerr=c,capsize=5, color='red', label='Explicit Distance Calculation')

a,b,c=plot_all('Decision_Tree/10_runs','Decision Tree')
plt.errorbar(a,b, yerr=c, capsize=5, color='blue',label='Decision Tree')

a,b,c=plot_all('QDA/10_runs','QDA')
plt.errorbar(a,b, yerr=c,capsize=5,color='green',label='QDA')

a,b,c=plot_all('Naive_Bayes/10_runs','Naive Bayes')
plt.errorbar(a,b, yerr=c,capsize=5, color='violet',label='Naive Bayes')

a,b,c=plot_all('Grad_Boost/10_runs','Grad Boost')
plt.errorbar(a,b, yerr=c,capsize=5,color='black',label='Grad Boost')


plt.legend(loc=4, fontsize=15)
#plt.xlabel(r"Volume Fraction  $(V_{f})$",fontsize=15)
plt.xlabel(r"Reduced Density $(\rho^{*})$",fontsize=15)
plt.ylabel(r"Nematic Order Parameter $(S)$",fontsize=15)
plt.tick_params(direction='in', length=6, width=2, colors='black', labelsize=15, grid_color='black')
plt.tight_layout()
plt.savefig('Nematic_Order_Parameter.png', dpi=200)
plt.show()










    
