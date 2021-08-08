# Import libraries
import numpy as np
import pandas as pd
import time
# Upload the dataset
diamonds = pd.read_csv('adult.data')
diamonds.head()

def unique_rows(check_unique):
    check_unique = np.ascontiguousarray(check_unique)
    unique = np.unique(check_unique.view([('', check_unique.dtype)]*check_unique.shape[1]))
    return unique.view(check_unique.dtype).reshape((unique.shape[0], check_unique.shape[1]))


print(diamonds)

# Import label encoder
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

diamonds = diamonds.drop([], axis = 1)
categorical_features = ['workclass','education','marital-status','occupation','relationship'
                        ,'race','sex','native-country','class',]
le = LabelEncoder()

# Convert the variables to numerical
for i in range(9):
    new = le.fit_transform(diamonds[categorical_features[i]])
    diamonds[categorical_features[i]] = new
diamonds.head()
#print(diamonds)
# Create features and target
X = diamonds[['age','workclass','education','education-num','marital-status','occupation','relationship'
                        ,'race','sex','capital-gain','capital-loss','hours-per-week','native-country','fnlwgt']]
y = diamonds[['class']]



# Make necessary imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
#print(y_train)
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

start = time.time()
# Train the model
regr = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=50, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=4, min_samples_split=8,
            min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=1,
            oob_score=True, random_state=None, verbose=0,
            warm_start=False)
regr.fit(X_train, y_train)

import warnings
warnings.filterwarnings('ignore')
y_pred = regr.predict(X_test)

feature_imp = pd.Series(regr.feature_importances_).sort_values(ascending=False)
#print(diamonds.loc[[0]])
#print(feature_imp)
end = time.time()


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print('oob: ', regr.oob_score_, ' ==> ', (1-regr.oob_score_)*100)
print('calculation time: ', end - start)

from sklearn.metrics import matthews_corrcoef
print('matthews: ',matthews_corrcoef(y_test, y_pred))

from sklearn.metrics import precision_score
print('precision score micro: ',precision_score(y_test, y_pred, average='micro'))
print('precision score macro: ',precision_score(y_test, y_pred, average='macro'))
