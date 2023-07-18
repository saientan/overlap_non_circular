import pylab
import numpy as np
import matplotlib.pyplot as plt

cutoff=0.11111111111111116

data=pylab.loadtxt('coord.dat')
x=data[:,0]
y=data[:,1]
z=data[:,2]
f=open('testing_data.dat','w')

def translate(x,y,dist_x,dist_y):
    xnew=x+dist_x
    ynew=y+dist_y
    return xnew, ynew


def translate_3d(x,y,z,dist_x,dist_y,dist_z):
    xnew=x+dist_x
    ynew=y+dist_y
    znew=z+dist_z
    return xnew, ynew, znew



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


def rotate_3d(x,y,z,alpha,beta,gamma):
    cmx=sum(x)/len(x)
    cmy=sum(y)/len(y)
    cmz=sum(z)/len(z)

    xr=x-cmx
    yr=y-cmy
    zr=z-cmz

    xrnew=(xr*np.cos(beta)*np.cos(gamma))+(yr*((np.sin(alpha)*np.sin(beta)*np.cos(gamma))-(np.cos(alpha)*np.sin(gamma))))+(zr*((np.cos(alpha)*np.sin(beta)*np.cos(gamma))+(np.sin(alpha)*np.sin(gamma)))) 

    yrnew=(xr*np.cos(beta)*np.sin(gamma))+(yr*((np.sin(alpha)*np.sin(beta)*np.sin(gamma))+(np.cos(alpha)*np.cos(gamma))))+(zr*((np.cos(alpha)*np.sin(beta)*np.sin(gamma))-(np.sin(alpha)*np.cos(gamma))))

    zrnew=(-xr*np.sin(beta))+(yr*np.sin(alpha)*np.cos(beta))+(zr*np.cos(alpha)*np.cos(beta))

    xnew=xrnew+cmx
    ynew=yrnew+cmy
    znew=zrnew+cmz

    return xnew, ynew, znew


def cal_overlap(x,y,xnew,ynew):
    overlap=0
    for i in range(0,len(x)):
        for j in range(0,len(xnew)):
            d=((x[i]-xnew[j])**2+(y[i]-ynew[j])**2)**0.5
            if d<=cutoff:
                overlap=1
                break
    return overlap


def cal_overlap_3d(x,y,z,xnew,ynew,znew):
    overlap=0
    for i in range(0,len(x)):
        for j in range(0,len(xnew)):
            d=((x[i]-xnew[j])**2+(y[i]-ynew[j])**2+(z[i]-znew[j])**2)**0.5
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
    alpha1=np.random.uniform(0,2*np.pi)
    beta1=np.random.uniform(0,np.pi)
    gamma1=np.random.uniform(0,2*np.pi)
    xnew, ynew, znew=rotate_3d(x,y,z,alpha1,beta1,gamma1)
    dist_x1=np.random.uniform(0,1)
    dist_y1=np.random.uniform(0,1)
    dist_z1=np.random.uniform(0,1)
    xnew1, ynew1,znew1=translate_3d(xnew, ynew,znew,dist_x1,dist_y1,dist_z1)

    #energy=cal_energy(x,y,xnew,ynew)
    #overlap=cal_overlap(x,y,xnew,ynew)
    
    alpha2=np.random.uniform(0,2*np.pi)
    beta2=np.random.uniform(0,np.pi)
    gamma2=np.random.uniform(0,2*np.pi)
    xnew, ynew, znew=rotate_3d(x,y,z,alpha2,beta2,gamma2)
    dist_x2=np.random.uniform(0,1)
    dist_y2=np.random.uniform(0,1)
    dist_z2=np.random.uniform(0,1)
    xnew2, ynew2,znew2=translate_3d(xnew, ynew,znew,dist_x2,dist_y2,dist_z2)


    overlap=cal_overlap_3d(xnew1,ynew1,znew1,xnew2,ynew2,znew2)

    diff_x=dist_x1-dist_x2
    diff_y=dist_y1-dist_y2
    diff_z=dist_z1-dist_z2
    diff_alpha=alpha1-alpha2
    diff_beta=beta1-beta2
    diff_gamma=gamma1-gamma2

    #plt.scatter(x,y,color='red')
    #plt.scatter(xnew,ynew,color='blue')
    #plt.title('overlap='+str(overlap))
    #plt.savefig('combine_figure.png')
    #plt.xlim(0,300)
    #plt.ylim(0,300)
    #plt.savefig('combine_figure.png')
    if overlap==0:
        count_0=count_0+1
        if count_0<501:
            f.write(str(diff_x)+'   '+str(diff_y)+'   '+str(diff_z)+'   '+str(diff_alpha)+'   '+str(diff_beta)+'   '+str(diff_gamma)+'   '+str(overlap)+'\n')
            f.write(str(-diff_x)+'   '+str(-diff_y)+'   '+str(-diff_z)+'   '+str(-diff_alpha)+'   '+str(-diff_beta)+'   '+str(-diff_gamma)+'   '+str(overlap)+'\n')
    if overlap==1:
        count_1=count_1+1
        if count_1<501:
            f.write(str(diff_x)+'   '+str(diff_y)+'   '+str(diff_z)+'   '+str(diff_alpha)+'   '+str(diff_beta)+'   '+str(diff_gamma)+'   '+str(overlap)+'\n')
            f.write(str(-diff_x)+'   '+str(-diff_y)+'   '+str(-diff_z)+'   '+str(-diff_alpha)+'   '+str(-diff_beta)+'   '+str(-diff_gamma)+'   '+str(overlap)+'\n')

    print (i, count_0, count_1)
    if count_0 > 500 and count_1> 500:
        break

    #if energy<1000:
     #   f.write(str(dist_x)+'   '+str(dist_y)+'   '+str(theta)+'   '+str(energy)+'\n')
    #plt.show()
    


