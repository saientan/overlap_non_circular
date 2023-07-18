import numpy as np
import matplotlib.pyplot as plt
import pylab
#N=149
#M=10

density=pylab.loadtxt('./1/density_vs_nematic_order_parameters.dat')
density=density[:,0]
N=len(density)

avg_order=[]
std_order=[]
for i in range(0,N):
    order=[]
    for j in range(1,11):
        data=pylab.loadtxt('./'+str(j)+'/density_vs_nematic_order_parameters.dat')
        order.append(data[:,1][i])
    avg=np.average(order)
    std=np.std(order)
    avg_order.append(avg)
    std_order.append(std)



plt.errorbar(density,avg_order,yerr=std_order, fmt='o-', markersize=8, capsize=10, label='Explicit distance calculation')
plt.show()










    
