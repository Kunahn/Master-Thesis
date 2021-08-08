# Import libraries
import numpy as np
import pandas as pd

# Upload the dataset
diamonds = pd.read_csv('bank-full.csv')
diamonds.head()

def unique_rows(check_unique):
    check_unique = np.ascontiguousarray(check_unique)
    unique = np.unique(check_unique.view([('', check_unique.dtype)]*check_unique.shape[1]))
    return unique.view(check_unique.dtype).reshape((unique.shape[0], check_unique.shape[1]))


#print(diamonds)

# Import label encoder
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D



diamonds = diamonds.drop(['contact','previous','poutcome','day','month','campaign','pdays'], axis = 1)
categorical_features = ['job', 'marital', 'education','default', 'housing', 'loan','y']
le = LabelEncoder()

# Convert the variables to numerical
for i in range(7):
    new = le.fit_transform(diamonds[categorical_features[i]])
    diamonds[categorical_features[i]] = new
#diamonds.head()


#print(diamonds)

# Create features and target
X = diamonds[['age','job', 'marital', 'education','default','balance', 'housing','y','duration',]]
y = diamonds[['loan']]
print(diamonds.iloc[0])
#print(diamonds)

print(diamonds.iloc[0])
# Make necessary imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Train the model
regr = RandomForestClassifier(n_estimators = 10, max_depth = 10, random_state = 101)
regr.fit(X_train, y_train)

import warnings
warnings.filterwarnings('ignore')
y_pred = regr.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))