import numpy as np
import pylab

def do_all(name):
    a=[]
    for i in range(1,11):
        a.append(pylab.loadtxt('time_taken_'+str(name)+'_'+str(i)))


    avg=np.average(a)
    std=np.std(a)
    print(avg,std)


do_all("NORMAL")
do_all("Decision_Tree")
do_all("QDA")
do_all("Naive_Bayes")
do_all("Grad_Boost")
