# Importing Image from PIL package
from PIL import Image
import matplotlib.pyplot as plt
import pylab
# creating a image object
im = Image.open(r"small_circle.png")
#px = im.load()
#print (px[4, 4])
#px[4, 4] = (0, 0, 0)
#print (px[4, 4])


plt.figure(figsize=(6, 6))

a=[]
b=[]


for i in range(0,100):
    for j in range(0,100):
        coordinate = x, y = i, j

        #print (i,j)
        # using getpixel method
        pix=im.getpixel(coordinate)
        #print (i,j,pix)
        #if pix!=(0, 0, 0, 255):
        if pix!=(255, 255, 255):
            a.append(i)
            b.append(j)
            #print (i,j)

r_a = max(a)-min(a)
r_b = max(b)-min(b)

for i in range(0,len(a)):
    a[i]=a[i]/63
    b[i]=b[i]/63

cmx=sum(a)/len(a)
cmy=sum(b)/len(b)


for i in range(0,len(a)):
    a[i]=a[i]-cmx
    b[i]=b[i]-cmy
    #print (a[i], b[i])
plt.scatter(a,b,color='blue',s=1000, alpha=0.05)

plt.xlabel("X",fontsize=15)
plt.ylabel("Y",fontsize=15)
plt.tick_params(direction='in', length=6, width=2, colors='black', labelsize=15, grid_color='black')
plt.tight_layout()
#plt.legend(fontsize=15)
#plt.savefig('Figure_traiangle.png', dpi=200)
#plt.show()



#a=[]
#b=[]
#for i in range(0,100,7):
#    for j in range(0,100,7):
#        coordinate = x, y = i, j

        #print (i,j)
        # using getpixel method
#        pix=im.getpixel(coordinate)
        #print (i,j,pix)
#        #if pix!=(0, 0, 0, 255):
#        if pix!=(255, 255, 255):
#            a.append(i)
#            b.append(j)
            #print (i,j)

#r_a = max(a)-min(a)
#r_b = max(b)-min(b)

#for i in range(0,len(a)):
#    a[i]=a[i]/63
#    b[i]=b[i]/63

#cmx=sum(a)/len(a)
#cmy=sum(b)/len(b)


data=pylab.loadtxt('coord.dat')
a=data[:,0]
b=data[:,1]


#for i in range(0,len(a)):
#    a[i]=a[i]-cmx
#    b[i]=b[i]-cmy
#    print (a[i], b[i])
plt.scatter(a,b,color='red', s=1000)


plt.xlabel("X",fontsize=15)
plt.ylabel("Y",fontsize=15)
plt.tick_params(direction='in', length=6, width=2, colors='black', labelsize=15, grid_color='black')
#plt.tight_layout()
#plt.legend(fontsize=15)
#plt.savefig('Figure_traiangle.png', dpi=200)
plt.xlim(-0.6,0.6)
plt.ylim(-0.6,0.6)
plt.tight_layout()
#plt.legend(fontsize=15)
plt.savefig('circle.png', dpi=200)

plt.show()



