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

classifiers = [
#    KNeighborsClassifier(3),
#    SVC(kernel="linear", C=0.025),
#    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
#    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#    MLPClassifier(alpha=1, max_iter=1000),
#    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    GradientBoostingClassifier(),
#    HistGradientBoostingClassifier(),
]


for something in range (0,len(names)):
    print (names[something])
    start= (time.time())


    #from __future__ import division
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    import pylab
    import os
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    import sklearn.linear_model
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.neural_network import MLPClassifier
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import HistGradientBoostingClassifier


    def learning_curve(training_data_location,testing_data_location,label):
        a=[]
        b=[]
        N=400   
        for i in range(1,N):
            j=25*i
            s1=training_data_location
            s="head -"+str(j)+" "+s1+" > learning_data.dat"
            #print (s)
            os.system(s)
    

            data_ML=pylab.loadtxt('learning_data.dat')
            x_train=data_ML[:,0:3]
            y_train=data_ML[:,3]
        #y_train=y_train.reshape(-1, 1)
            data_ML=pylab.loadtxt(testing_data_location)
            x_test=data_ML[:,0:3]
            y_test=data_ML[:,3]
            y_test=y_test.reshape(-1, 1) 
        #model = GradientBoostingClassifier()
        #model = GradientBoostingClassifier(max_depth= 10, min_samples_leaf= 5, min_samples_split= 10, n_estimators = 50)
            #model=sklearn.ensemble.HistGradientBoostingClassifier()
        #model = RandomForestClassifier()
            model=classifiers[something] 
            model.fit(x_train,y_train)

            pred_y_test=model.predict(x_test)

            accuracy=0
            for k in range(0,len(y_test)):
                if y_test[k]==pred_y_test[k]:
                    accuracy=accuracy+1
            accuracy=(accuracy*100)/(len(y_test))
            a.append(j)
            b.append(accuracy)
      
    #a=np.log10(a)  
    #plt.scatter(a,b)
    #plt.plot(a,b)
        return a,b,label



    a,b,label=learning_curve("./Circle/new_training_data.dat","./Circle/testing_data.dat",'Circle')
    #print(a,b)
    np.savetxt('acc_Circle_'+str(names[something]),b)
    np.savetxt('N_Circle_'+str(names[something]),a)

    plt.scatter(a,b,label=label)

    plt.plot(a,b)

    a,b,label=learning_curve("./Triangle/new_training_data.dat","./Triangle/testing_data.dat", 'Triangle')
    np.savetxt('acc_Triangle_'+str(names[something]),b)
    np.savetxt('N_Triangle_'+str(names[something]),a)

    plt.scatter(a,b,label=label)
    plt.plot(a,b)
    a,b,label=learning_curve("./Rectangle/new_training_data.dat","./Rectangle/testing_data.dat",'Rod')
    np.savetxt('acc_Rectangle_'+str(names[something]),b)
    np.savetxt('N_Rectangle_'+str(names[something]),a)
    plt.scatter(a,b,label=label)
    plt.plot(a,b)

    a,b,label=learning_curve("./Star/new_training_data.dat","./Star/testing_data.dat",'Star')
    np.savetxt('acc_Star_'+str(names[something]),b)
    np.savetxt('N_Star_'+str(names[something]),a)
    plt.scatter(a,b,label=label)
    plt.plot(a,b)

    plt.legend(fontsize=15)

    plt.xscale('log',base=10) 


    plt.xlabel("Number of training data",fontsize=15)
    plt.ylabel("Prediction accuracy (%)",fontsize=15)
    plt.tick_params(direction='in', length=6, width=2, colors='black', labelsize=15, grid_color='black')
    plt.title(names[something],fontsize=15)
    plt.tight_layout()
    plt.savefig('Figure_'+str(names[something])+'.png', dpi=200)
    #plt.show()
    plt.close()
    end= (time.time())
    print ("time_taken", end-start)







