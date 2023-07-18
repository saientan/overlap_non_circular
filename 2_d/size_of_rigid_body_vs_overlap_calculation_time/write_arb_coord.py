f=open('coord.dat','w')
N=14
for i in range(0,14):
    f.write('0.0   '+str(-i*0.11)+'\n')
f.write('0.0 1.1102230246251565e-16'+'\n')
for i in range(0,14):
    f.write('0.0   '+str(i*0.11)+'\n')

