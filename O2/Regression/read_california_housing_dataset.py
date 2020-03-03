# Read California Housing data
# https://scikit-learn.org/dev/datasets/index.html#california-housing-dataset
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

from sklearn.datasets.california_housing import fetch_california_housing
import matplotlib.pyplot as plt

cal_housing = fetch_california_housing()
X, y = cal_housing.data, cal_housing.target
names = cal_housing.feature_names

#%% Show example data

Nsamples = X.shape[0]

n = 42
xn = X[n, :]

# Input values
for i,val in enumerate(xn):
    print(names[i], val)

# Target value
print('Target :', y[n])

#%% 1D data plot

plt.plot(X[:,0], y, '.', markersize=1)
plt.title(names[0])

    
