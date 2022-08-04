# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import necessary modules
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers, utils, backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.datasets import  make_classification
from sklearn.ensemble import RandomForestClassifier

def plot_roc_curve(y_test, y_pred):
  
  n_classes = len(np.unique(y_test))
  y_test = label_binarize(y_test, classes=np.arange(n_classes))
  y_pred = label_binarize(y_pred, classes=np.arange(n_classes))

  # Compute ROC curve and ROC area for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
  
  # Compute micro-average ROC curve and ROC area
  fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

  # First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

  # Finally average it and compute AUC
  mean_tpr /= n_classes

  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  # Plot all ROC curves
  plt.figure(figsize=(10,5))
  #plt.figure(dpi=600)
  lw = 2
  plt.plot(fpr["micro"], tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink", linestyle=":", linewidth=4,)

  plt.plot(fpr["macro"], tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy", linestyle=":", linewidth=4,)

  colors = cycle(["aqua", "darkorange", "darkgreen", "yellow", "blue"])
  for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),)

  plt.plot([0, 1], [0, 1], "k--", lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("Receiver Operating Characteristic (ROC) curve")
  plt.legend()
  plt.savefig('Experiment ROC.png')
  plt.show()

filepath = 'Dataset.csv'
data = pd.read_csv(filepath, delimiter=';', header=1)
writePath = 'results.txt'
data.columns = range(data.shape[1])

#Get number of input features from data set 
features = data.shape[1]

#Get number of unique labels from data set 
data_labels = data[features-1].unique()


#Get the number of unique labels in the dataset
length = len(data_labels)

#Copy data to X - labels and labels to y
#Changing pandas dataframe to numpy array
X = data.iloc[:,:features-1].values
y = data.iloc[:,features-1:features].values

#Normalizing the data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)


#One hot encoding
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()



indices = np.arange(len(X))

#Split the data in training and testing, 75% as training and 25% as training
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test,indices_train, indices_test = train_test_split(X,y,indices, random_state=42)

# Neural network
model = Sequential()
# 9/4 eller 16/8
model.add(Dense(9, input_dim=features-1, activation='relu'))
model.add(Dense(4, activation='relu'))
#model.add(Dropout(0.01))
model.add(Dense(length, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Print and save model design
print(model.summary())
utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

#Train model 
training = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_split=0.2)

#Save model 
model.save('model.h5')

#Evaluate model
score = model.evaluate(x_train, y_train, batch_size=32)

#Print(score)
calc_loss = "%s: %.2f%%" % (model.metrics_names[0], score[0]*100)
calc_acc = "%s: %.2f%%" % (model.metrics_names[1], score[1]*100)
print(calc_loss)
print(calc_acc) 

#Predicting the Test set rules
#Greater than 0.50 on scale 0 to 1
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5) 


#Plot the training and testing accuracy and loss at each epoch
loss = training.history['loss']
val_loss = training.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper right')
plt.savefig('Experiment loss.png')
plt.show()


acc = training.history['accuracy']
val_acc = training.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='lower right')
plt.savefig('Experiment acc.png')
plt.show()


from sklearn.metrics import precision_score



#Making confusion matrix that checks accuracy of the model.
import warnings

#Supress warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

from sklearn.metrics import classification_report
cr = classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), labels=data_labels)
print('Classification report : \n',cr)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1), labels=data_labels)
print(cm)


FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
print("FP: ", FP,"\nFN: ",FN,"\nTP: ",TP,"\nTN: ",TN)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
TPR = np.round(TPR,4)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
TNR = np.round(TNR,4)
# Precision or positive predictive value
PPV = TP/(TP+FP)
PPV = np.round(PPV,4)
# Negative predictive value
NPV = TN/(TN+FN)
NPV = np.round(NPV,4)
# Fall out or false positive rate
FPR = FP/(FP+TN)
FPR = np.round(FPR,4)
# False negative rate
FNR = FN/(TP+FN)
FNR = np.round(FNR,4)
# False discovery rate
FDR = FP/(TP+FP)
FDR = np.round(FDR,4)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
ACC = np.round(ACC,4)

print("TPR: ", TPR,"\nTNR: ",TNR,"\nNPV: ",NPV,"\nFPR: ",FPR,"\nFNR: ",FNR,"\nFDR: ",FDR,"\nACC: ",ACC)

#Plot confusion matrix
ax = sns.heatmap(cm, annot=True, fmt='', cbar=False, cmap='Blues')
#ax.set_title('Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
h,l = ax.get_legend_handles_labels()
ax.legend(h,l, borderaxespad=0)
#ax.axis("off")
plt.tight_layout()
plt.savefig('Experiment CM.png')
plt.show()

import io
s = io.StringIO()
model.summary(print_fn=lambda x: s.write(x + '\n'))
model_summary = s.getvalue()
s.close()

# Open a file with access mode 'a'
file_object = open(writePath, 'a')
file_object.write(filepath)
file_object.write("\n")
file_object.write(model_summary)
file_object.write("\n")
file_object.write(calc_loss)
file_object.write("\n")
file_object.write(calc_acc)
file_object.write("\n")
file_object.write('\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))
file_object.write('\nFP: {}\nFN: {}\nTP: {}\nTN: {}\n'.format(FP, FN, TP, TN))
file_object.write('\nFPR: {}\nFNR: {}\nTPR: {}\nTNR: {}\n'.format(FPR, FNR, TPR, TNR))
file_object.write('\nNPV: {}\nFDR: {}\nACC: {}\n'.format(NPV, FDR, ACC))
file_object.write("\n\n---------------------------------------------\n\n")
#dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

# Close the file
file_object.close()

#ROC plot
plot_roc_curve(y_test.argmax(axis=1), y_pred.argmax(axis=1))
 

print("END")