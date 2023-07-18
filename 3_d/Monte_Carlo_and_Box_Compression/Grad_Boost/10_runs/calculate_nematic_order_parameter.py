from __future__ import division
import matplotlib.pyplot as plt
f=open('for_vmd_ML.xyz','r')
l_f=f.readlines()


def order(number):

    nmol=64
    natom=9
    nframe=98

    cos2_theta=0
    for frame in range(number*98,(number+1)*98):
        a=(((nmol*natom)+2)*frame)+2
        b=a+(nmol*natom)
        for i in range(a,b,natom):   
            start=i
            end=i+(natom-1)
            #print (start,end)
            vx=float(l_f[end].split()[1])-float(l_f[start].split()[1])
            vy=float(l_f[end].split()[2])-float(l_f[start].split()[2])
            vz=float(l_f[end].split()[3])-float(l_f[start].split()[3])
            r=(vx**2+vy**2+vz**2)**0.5
            cos2_theta=(vy/r)**2+cos2_theta

    cos2_theta=cos2_theta/(nframe*nmol)
    return ((1.5*cos2_theta)-0.5)


plt.figure(figsize=(7.5, 5.5))


a=[]
b=[]
N=64
phi=0.25
L=(N/phi)**(1/3)


for i in range(1,200):
    a.append(order(i))
    #print (64/L**3)
    b.append(64/(L**3))
    #L=L-((4)/(3*L*L)) 
    L=(64/((64/(L**3))+0.075))**(1/3)
    #print (64/L**3)
    #L=L-0.1import matplotlib.pyplot as pli

#plt.plot(b)
#plt.show()

#print(b)
#plt.plot(b,a)

f=open('density_vs_nematic_order_parameters.dat','w')
for i in range(0,len(a)):
    f.write(str(b[i])+'   '+str(a[i])+'\n')
f.close()

#plt.plot(b,a, '-o',color='red', label='Explicit Distance Calculation')

#plt.show()
'''
f=open('./Decision_Tree/for_vmd_ML.xyz','r')
l_f=f.readlines()
a=[]
b=[]
N=64
phi=0.25
L=(N/phi)**(1/3)


for i in range(1,200):
    a.append(order(i))
    b.append(64/(L**3))
    #L=L-((4)/(3*L*L))
    L=(64/((64/(L**3))+0.075))**(1/3)
#print(a)
#plt.plot(b,a)
plt.plot(b,a, '-o', color='blue',label='Decision Tree')

f=open('./QDA/for_vmd_ML.xyz','r')
l_f=f.readlines()
a=[]
b=[]
N=64
phi=0.25
L=(N/phi)**(1/3)


for i in range(1,200):
    a.append(order(i))
    b.append(64/(L**3))
    #L=L-((4)/(3*L*L))
    L=(64/((64/(L**3))+0.075))**(1/3)

#print(a)
#plt.plot(b,a)
plt.plot(b,a, '-o', color='green',label='QDA')

f=open('./Naive_Bayes/for_vmd_ML.xyz','r')
l_f=f.readlines()
a=[]
b=[]
N=64
phi=0.25
L=(N/phi)**(1/3)


for i in range(1,200):
    a.append(order(i))
    b.append(64/(L**3))
    #L=L-((4)/(3*L*L))
    L=(64/((64/(L**3))+0.075))**(1/3)
#print(a)
#plt.plot(b,a)

plt.plot(b,a, '-o', color='violet',label='Naive Bayes')

f=open('./Grad_Boost/for_vmd_ML.xyz','r')
l_f=f.readlines()
a=[]
b=[]
N=64
phi=0.25
L=(N/phi)**(1/3)


for i in range(1,200):
    a.append(order(i))
    b.append(64/(L**3))
    #L=L-((4)/(3*L*L))
    L=(64/((64/(L**3))+0.075))**(1/3)
#print(a)
#plt.plot(b,a)

plt.plot(b,a, '-o', color='black',label='Grad Boost')
print (b)
plt.legend(loc=4, fontsize=15)
plt.xlabel(r"Number Density $(\phi)$",fontsize=15)
plt.ylabel(r"Nematic Order Parameter $(S)$",fontsize=15)
plt.tick_params(direction='in', length=6, width=2, colors='black', labelsize=15, grid_color='black')
plt.tight_layout()
plt.savefig('Nematic_Order_Parameter.png', dpi=200)

#plt.plot(a)
plt.show()

'''
