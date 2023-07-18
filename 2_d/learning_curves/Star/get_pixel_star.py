# Importing Image from PIL package
from PIL import Image
import matplotlib.pyplot as plt
# creating a image object
im = Image.open(r"star_inside_small_circle.png")
#px = im.load()
#print (px[4, 4])
#px[4, 4] = (0, 0, 0)
#print (px[4, 4])

a=[]
b=[]
for i in range(0,100,7):
    for j in range(0,100,7):
        coordinate = x, y = i, j
 
        #print (i,j)
        # using getpixel method
        pix=im.getpixel(coordinate)
        #print (i,j,pix)
        #if pix!=(0, 0, 0, 255):
        if pix==(237, 28, 36, 255): 
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
    print (a[i], b[i])
plt.scatter(a,b)
plt.xlim(-2,2)
plt.ylim(-2,2)

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
        if pix==(237, 28, 36, 255):
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

#g=open('full_star_coord.dat','w')

for i in range(0,len(a)):
    a[i]=a[i]-cmx
    b[i]=b[i]-cmy
#    g.write(str(a[i])+'   '+str(b[i])+'\n')
plt.scatter(a,b,color='red', alpha=0.1)

#g.close()
plt.show()




