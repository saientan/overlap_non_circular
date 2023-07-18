import pylab
import matplotlib.pyplot as plt



data=pylab.loadtxt('time_taken.dat')


plt.plot(data[:,0], data[:,1], label="NORMAL")
plt.plot(data[:,0], data[:,2], label="ML")

plt.legend()
plt.show()

