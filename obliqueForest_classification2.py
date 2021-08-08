
# Import libraries
import numpy as np
import pandas as pd
import time
# Upload the dataset
diamonds = pd.read_csv('bank-full.csv')
diamonds.head()

def unique_rows(check_unique):
    check_unique = np.ascontiguousarray(check_unique)
    unique = np.unique(check_unique.view([('', check_unique.dtype)]*check_unique.shape[1]))
    return unique.view(check_unique.dtype).reshape((unique.shape[0], check_unique.shape[1]))

#print(diamonds)

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn_oblique_tree.oblique import ObliqueTree
# Import label encoder
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


diamonds = diamonds.drop(['contact','previous','poutcome','day','month','campaign','pdays','y','default','marital', 'education', 'housing','job' ], axis = 1)
categorical_features = ['loan']
le = LabelEncoder()

# Convert the variables to numerical
for i in range(1):
    new = le.fit_transform(diamonds[categorical_features[i]])
    diamonds[categorical_features[i]] = new
diamonds.head()

# Create features and target
X = diamonds[['age','balance', 'duration',]]
y = diamonds[['loan']]

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=101) # 70% training and 30% test

#Import svm model
from sklearn import svm
start = time.time()
#Create a svm Classifier
clf = ObliqueTree(splitter="oc1", number_of_restarts=20, max_perturbations=5, random_state=101)

#Train the model using the training sets
clf.fit(X_train, y_train)

import warnings
warnings.filterwarnings('ignore')
#Predict the response for test dataset
y_pred = clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
end = time.time()

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print('calculation time: ', end - start)

from sklearn.metrics import matthews_corrcoef
print('matthews: ',matthews_corrcoef(y_test, y_pred))

from sklearn.metrics import precision_score
print('precision score micro: ',precision_score(y_test, y_pred, average='micro'))
print('precision score macro: ',precision_score(y_test, y_pred, average='macro'))