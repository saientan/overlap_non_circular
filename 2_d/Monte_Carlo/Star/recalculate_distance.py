from __future__ import division
import pylab

data=pylab.loadtxt('initial_snap.dat')
x=data[:,0]
y=data[:,1]

l=len(x)
d=[]
for i in range(0,l):
    print (i, l)
    for j in range(0,l):
        if i!=j:
            dist=((x[i]-x[j])**2+(y[i]-y[j])**2)**0.5
            d.append(dist)

print (min(d))





