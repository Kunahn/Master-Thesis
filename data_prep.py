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

print(diamonds)
#diamonds.to_csv (r'I:\documents\PythonProjects\master\bank-prep.csv', header=True)