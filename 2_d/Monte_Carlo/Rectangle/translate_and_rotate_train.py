import pylab
import numpy as np
import matplotlib.pyplot as plt



cutoff=0.11111111111111116

data=pylab.loadtxt('coord.dat')
x=data[:,0]
y=data[:,1]

f=open('training_data.dat','w')


def translate(x,y,dist_x,dist_y):
    xnew=x+dist_x
    ynew=y+dist_y
    return xnew, ynew

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


def cal_overlap(x,y,xnew,ynew):
    overlap=0
    for i in range(0,len(x)):
        for j in range(0,len(xnew)):
            d=((x[i]-xnew[j])**2+(y[i]-ynew[j])**2)**0.5
            if d<=cutoff:
                overlap=1
                break
    return overlap

def cal_energy(x,y,xnew,ynew):
    energy=0
    for i in range(0,len(x)):
        for j in range(0,len(xnew)):
            d=((x[i]-xnew[j])**2+(y[i]-ynew[j])**2)**0.5
            energy=energy+(((1/d)**12)-((1/d)**6))
            
    return energy

count_0=0
count_1=0
for i in range(0,10000000):
    #print (i)
    theta=np.random.uniform(0,2*np.pi)
    xnew, ynew=rotate(x,y,theta)
    dist_x=np.random.uniform(0,6)
    dist_y=np.random.uniform(0,6)
    xnew, ynew=translate(xnew, ynew,dist_x,dist_y)

    #energy=cal_energy(x,y,xnew,ynew)
    overlap=cal_overlap(x,y,xnew,ynew)
    #plt.scatter(x,y,color='red')
    #plt.scatter(xnew,ynew,color='blue')
    #plt.title('overlap='+str(overlap))
    #plt.savefig('combine_figure.png')
    #plt.xlim(0,300)
    #plt.ylim(0,300)
    #plt.savefig('combine_figure.png')
    if overlap==0:
        count_0=count_0+1
        if count_0<5001:
            f.write(str(dist_x)+'   '+str(dist_y)+'   '+str(theta)+'   '+str(overlap)+'\n')
    if overlap==1:
        count_1=count_1+1
        if count_1<5001:
            f.write(str(dist_x)+'   '+str(dist_y)+'   '+str(theta)+'   '+str(overlap)+'\n')
    print (i, count_0, count_1)
    if count_0 > 5000 and count_1> 5000:
        break

    #if energy<1000:
     #   f.write(str(dist_x)+'   '+str(dist_y)+'   '+str(theta)+'   '+str(energy)+'\n')
    #plt.show()
    


