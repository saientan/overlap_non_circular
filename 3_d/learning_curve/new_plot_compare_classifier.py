# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pylab
import time
names = [
#    "Nearest Neighbors",
#    "Linear SVM",
#    "RBF SVM",
    "Decision Tree",
#    "Random Forest",
#    "Neural Net",
#    "AdaBoost",
    "Naive Bayes",
    "QDA",
    "Grad Boost",
#    "HistGradientBoostingClassifier",
]


for something in range(0,len(names)):

    acc=pylab.loadtxt('acc_Rectangle_'+str(names[something]))
    N=pylab.loadtxt('N_Rectangle_'+str(names[something]))

    plt.scatter(N,acc,label=names[something])

    plt.plot(N,acc)

    
    plt.legend(fontsize=15)

    plt.xscale('log',base=10) 


    plt.xlabel("Number of training data",fontsize=15)
    plt.ylabel("Prediction accuracy (%)",fontsize=15)
    plt.tick_params(direction='in', length=6, width=2, colors='black', labelsize=15, grid_color='black')
    #plt.title(names[something],fontsize=15)
    #plt.ylim(65,100)
    plt.tight_layout()



plt.title('Rod',fontsize=15)
plt.tight_layout()
plt.savefig('Figure_all_learning_Rod.png', dpi=200)
plt.show()
plt.close()
    







