import pylab
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

nd = [9, 31, 32, 67]
name_sys = ['Rod', 'Star', 'Triangle', 'Circle']
start = [0.12, 0.72, 0.66, 0.98] #initial r value for rod, star, triangle and circle respectively

name = name_sys[0]
num_disk = nd[0]
initial = start[0]

def orient_corref(M, N, NP, L, bin_width, name, sys, initial):

    r = np.arange(initial, L/2, bin_width)
    r2 = []

    for run in range(0,10):
        temp = []
        for l in range(len(r)-1):
            a1 = []
            a2 = []
            for i in range(M,N):
                data = pylab.loadtxt('./' + sys + '/' + str(run) + '_' + name + '_' + str(round(L,2)) + '/traj_MC_' + name + '_' + str(round(L,2)) + '_' + str(i) + '.dat')
                x = data[:,0]
                y = data[:,1]
                theta = data[:,2]

                for j in range(0,NP):
                    for k in range(j,NP):
                        dx = abs(x[j]-x[k])
                        dy = abs(y[j]-y[k])

                        #periodic boundary condition
                        if dx>L/2: dx = L-dx
                        if dy>L/2: dy = L-dy

                        d = (dx**2 + dy**2)**0.5

                        t1 = theta[j]
                        if t1 >= np.pi:
                            t1 = t1 - np.pi
                            
                        t2 = theta[k]
                        if t2 >= np.pi:
                            t2 = t2 - np.pi

                        if d > r[l] and d < r[l+1]:
                            a1.append(t1)
                            a2.append(t2)

            correlation_matrix = np.corrcoef(a2, a1)
            corr=correlation_matrix[0, 1]
            temp.append(corr)
            
        r2.append(temp)
        # print(run)

    std = np.std(r2, axis=0)
    results = np.average(r2, axis=0)

    r = r[0:-1]

    return results, std, r

Area_frac = 0.20
D = 0.11111111111
phi = 4*Area_frac/(num_disk*np.pi*D*D)

M = 1
N = 100
NP = 64

L = (NP/phi)**0.5
n = NP/(L**2)

bin_width = 0.15

results2, std2, r = orient_corref(M, N, NP, L, bin_width, 'ML_Grad_Boost', name, initial)
results, std, r = orient_corref(M, N, NP, L, bin_width, 'NORMAL', name, initial)

plt.errorbar(r, results, yerr=std, ecolor='red', elinewidth=1, capsize=5, color='red', label='Explicit Distance Calculation')
plt.errorbar(r, results2, yerr=std2, ecolor='blue', elinewidth=1, capsize=5, color='blue', label='ML Model')

plt.title(name, fontsize=15)
plt.legend(loc=1, fontsize=15)
plt.ylabel(r"$R²$",fontsize=15)
plt.xlabel(r"$r$",fontsize=15)
plt.tick_params(direction='in', length=6, width=2, colors='black', labelsize=15, grid_color='black')

plt.tight_layout()
plt.savefig('./' + name + '_r_squared_up.png', dpi=200)
plt.show()