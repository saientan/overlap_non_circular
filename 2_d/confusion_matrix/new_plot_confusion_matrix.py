import numpy as np
import matplotlib.pyplot as plt
import pylab
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
def show_confusion_matrix(training_data_location,testing_data_location ,Name):

    class_names = ["Overlap", "No_overlap"]

# Split the data into a training set and a test set
    data_ML=pylab.loadtxt(training_data_location)
    x_train=data_ML[:,0:3]
    y_train=data_ML[:,3]
        #y_train=y_train.reshape(-1, 1)
    data_ML=pylab.loadtxt(testing_data_location)
    x_test=data_ML[:,0:3]
    y_test=data_ML[:,3]
    #y_test=y_test.reshape(-1, 1)
    model = GradientBoostingClassifier()
    model.fit(x_train,y_train)

    y_pred_test=model.predict(x_test)
    np.set_printoptions(precision=3)
    #plt.subplot(121)

#    disp = ConfusionMatrixDisplay.from_estimator(model,x_test,y_test,display_labels=class_names,cmap=plt.cm.Blues)

    #plt.title(Name)    
    #plt.show()
    return y_test, y_pred_test



#fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

ax = plt.subplot(221)

y_test, y_pred_test = show_confusion_matrix("../learning_curves/Circle/new_training_data.dat","../learning_curves/Circle/testing_data.dat","Circle")

cm= (confusion_matrix(list(y_test), list(y_pred_test), labels=[0.0, 1.0]))

sns.set(font_scale=1.5) # Adjust to fit
sns.heatmap(cm, annot=True, ax=ax, fmt="g");  

# Labels, title and ticks
label_font = {'size':'15'}  # Adjust to fit
ax.set_xlabel('Predicted', fontdict=label_font);
ax.set_ylabel('Actual', fontdict=label_font);
#ax.text(1,1,'(a)', size=15)
title_font = {'size':'15'}  # Adjust to fit
ax.set_title('Circle', fontdict=title_font);

ax.tick_params(axis='both', which='major', labelsize=15)  # Adjust to fit
ax.xaxis.set_ticklabels(['Overlap', 'No Overlap']);
ax.yaxis.set_ticklabels(['Overlap', 'No Overlap']);


ax = plt.subplot(222)

y_test, y_pred_test = show_confusion_matrix("../learning_curves/Triangle/new_training_data.dat","../learning_curves/Triangle/testing_data.dat","Triangle")

cm= (confusion_matrix(list(y_test), list(y_pred_test), labels=[0.0, 1.0]))

sns.set(font_scale=1.5) # Adjust to fit
sns.heatmap(cm, annot=True, ax=ax, fmt="g");

# Labels, title and ticks
label_font = {'size':'15'}  # Adjust to fit
ax.set_xlabel('Predicted', fontdict=label_font);
ax.set_ylabel('Actual', fontdict=label_font);

title_font = {'size':'15'}  # Adjust to fit
ax.set_title('Triangle', fontdict=title_font);

ax.tick_params(axis='both', which='major', labelsize=15)  # Adjust to fit
ax.xaxis.set_ticklabels(['Overlap', 'No Overlap']);
ax.yaxis.set_ticklabels(['Overlap', 'No Overlap']);


ax = plt.subplot(223)

y_test, y_pred_test = show_confusion_matrix("../learning_curves/Rectangle/new_training_data.dat","../learning_curves/Rectangle/testing_data.dat","Circle")

cm= (confusion_matrix(list(y_test), list(y_pred_test), labels=[0.0, 1.0]))

sns.set(font_scale=1.5) # Adjust to fit
sns.heatmap(cm, annot=True, ax=ax, fmt="g");

# Labels, title and ticks
label_font = {'size':'15'}  # Adjust to fit
ax.set_xlabel('Predicted', fontdict=label_font);
ax.set_ylabel('Actual', fontdict=label_font);

title_font = {'size':'15'}  # Adjust to fit
ax.set_title('Reactangle', fontdict=title_font);

ax.tick_params(axis='both', which='major', labelsize=15)  # Adjust to fit
ax.xaxis.set_ticklabels(['Overlap', 'No Overlap']);
ax.yaxis.set_ticklabels(['Overlap', 'No Overlap']);


ax = plt.subplot(224)

y_test, y_pred_test = show_confusion_matrix("../learning_curves/Star/new_training_data.dat","../learning_curves/Star/testing_data.dat","Star")

cm= (confusion_matrix(list(y_test), list(y_pred_test), labels=[0.0, 1.0]))

sns.set(font_scale=1.5) # Adjust to fit
sns.heatmap(cm, annot=True, ax=ax, fmt="g");

# Labels, title and ticks
label_font = {'size':'15'}  # Adjust to fit
ax.set_xlabel('Predicted', fontdict=label_font);
ax.set_ylabel('Actual', fontdict=label_font);

title_font = {'size':'15'}  # Adjust to fit
ax.set_title('Star', fontdict=title_font);

ax.tick_params(axis='both', which='major', labelsize=15)  # Adjust to fit
ax.xaxis.set_ticklabels(['Overlap', 'No Overlap']);
ax.yaxis.set_ticklabels(['Overlap', 'No Overlap']);

plt.tight_layout()
plt.show()









