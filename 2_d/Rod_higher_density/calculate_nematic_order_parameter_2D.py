import matplotlib.pyplot as plt
f=open('for_vmd_NORMAL_.xyz','r')
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
            cos2_theta=(vx/r)**2+cos2_theta

    cos2_theta=cos2_theta/(nframe*nmol)
    return ((1.5*cos2_theta)-0.5)
a=[]
for i in range(1,55):
    a.append(order(i))

print(a)
plt.plot(a)
#plt.show()
#f=open('for_vmd_ML.xyz','r')
#l_f=f.readlines()
#a=[]
#for i in range(1,22):
#    a.append(order(i))

#print(a)
#plt.plot(a)



#plt.scatter(a)
plt.show()


