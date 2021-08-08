# Import libraries
import numpy as np
import pandas as pd

# Upload the dataset
diamonds = pd.read_csv('bank2.csv')
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
#print(diamonds.iloc[0])


#diamonds = diamonds.drop(['Unnamed: 0'], axis = 1)
categorical_features = ['age', 'gender', 'abschluss']
le = LabelEncoder()

# Convert the variables to numerical
for i in range(3):
    new = le.fit_transform(diamonds[categorical_features[i]])
    diamonds[categorical_features[i]] = new
#diamonds.head()


#print(diamonds)

# Create features and target
X = diamonds[['age']]
y = diamonds[['abschluss']]

print(diamonds)

# Make necessary imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101)



# Train the model
regr = RandomForestRegressor(n_estimators = 10, max_depth = 10, random_state = 101)
regr.fit(X_train, y_train.values.ravel())

import warnings
warnings.filterwarnings('ignore')

# Make prediction
predictions = regr.predict(X_test)

result = X_test
result['price'] = y_test
result['prediction'] = predictions.tolist()
result.head()


# Import library for visualization
import matplotlib.pyplot as plt

# Define x axis
#x_axis = X_test.carat

nresult = np.array(result)
carat = nresult[:, 0]
#clarity = nresult[:, 6]
#cut = nresult[:, 7] 
#colorb = nresult[:, 8] 
#price = nresult[:, 1]
#nprediction = nresult[:, 2]

plt.figure(2, figsize=(8, 6))
plt.clf()
# Build scatterplot
plt.scatter(carat, y_test, c = 'b', alpha = 0.5, marker = '.', label = 'Real')
plt.scatter(carat, predictions, c = 'r', alpha = 0.5, marker = '.', label = 'Predicted')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.grid(color = '#D3D3D3', linestyle = 'solid')
plt.legend(loc = 'lower right')
#plt.show()

'''
# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
'''
nresult = np.array(result)
carat = nresult[:, 0]
#clarity = nresult[:, 6]
#cut = nresult[:, 7] 
#colorb = nresult[:, 8] 
price = nresult[:, 1]
#nprediction = nresult[:, 2]
#presult = np.concatenate((carat, clarity), axis = None)


# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(10, 8))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(result)
#ax.scatter(x_axis, predictions, c='r',
#           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=price,
           cmap=plt.cm.Set1, edgecolor='k', s=40)

##ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
##           cmap=plt.cm.Set1, edgecolor='k', s=40)

#ax.scatter(x_axis, y_test, c='b',
#           cmap=plt.cm.Set1, edgecolor='k', s=40)           
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

'''
print(presult)

print(nresult[:, 0])
print("------")
print("------")
#print(X_reduced)
print("------")
print("------")
#print(X_reduced[:, 0])
print("------")
#print(X_reduced[:, 1])
print("------")
#print(X_reduced[:, 2])
print("------")
'''

ny = np.array(y)
ny = unique_rows(ny)
#print(ny)

#print(np.split(carat, [1, 2]))
numOfRows = np. size(ny, 0)
#print(numOfRows)


plt.show()

# Import library for metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Mean absolute error (MAE)
mae = mean_absolute_error(y_test.values.ravel(), predictions)

# Mean squared error (MSE)
mse = mean_squared_error(y_test.values.ravel(), predictions)

# R-squared scores
r2 = r2_score(y_test.values.ravel(), predictions)

# Print metrics
print('BT Mean Absolute Error:', round(mae, 2))
print('BT Mean Squared Error:', round(mse, 2))
print('BT R-squared scores:', round(r2, 2))
