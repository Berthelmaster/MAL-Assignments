# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:51:45 2020

@author: Thomas Berthel
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.datasets.california_housing import fetch_california_housing
from math import sqrt
import matplotlib.pyplot as plt

housing_data = fetch_california_housing()
X_data, y_target = housing_data.data, housing_data.target
names = housing_data.feature_names

Nsamples = X_data.shape[0]
print(Nsamples)

#Vi sætter vores input variabler samt target
n = 42
xn = X_data[n, :]

for i,val in enumerate(xn):
    print(names[i], val)

print('Target :', y_target[n])

#Øvelse 1 - A -Plot “median house value for California districts” (target værdi)
# i forhold til “MedInc median income in block”

plt.plot(X_data[:,0], y_target, '.', markersize=1)
plt.ylabel('Median house value')
plt.xlabel('MedInc median income in block')
plt.show()

#Øvelse 1 - B 
#Først sørger vi for at lave et training og testing set
#Som begge er e 1/5 af dataen
X_train = X_data[:-4128]
X_test = X_data[-4128:]

#Og derefter af target
y_train = y_target[:-4128]
y_test = y_target[-4128:]


lin_reg = linear_model.LinearRegression()

# Inden vi kan lave en prædiktion så skal vi træne vores model ved brug af træningssættet
#Samt lave et reshap for at det passer sammen
X_MedInc = X_train[:,0].reshape(-1,1) 
lin_reg.fit(X_MedInc, y_train)

# Vi laver vores prædiktion ved brug af træningssæt
y_pred = lin_reg.predict(X_test[:,0].reshape(-1,1))
print(y_pred)

# The root mean squared error
rmse_MedInc = sqrt(mse(y_test,y_pred))
print(rmse_MedInc)
print("Som vi kan se er fejlen 0.8285")



residual = abs(y_test-y_pred)
plt.scatter(X_test[:,0].reshape(-1,1),residual)
plt.xlabel('MedInc median income in block')
plt.ylabel('forskellen på test og predict')

plt.hist(residual,bins=100)
plt.xlabel('forskellen på test og predict')
plt.ylabel('Feature samples count')

#Øvelse 1 - B - Prædikere hus priser udfra alle features
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

print(y_pred)

rmse = sqrt(mse(y_test,y_pred))

print("RMSE:")
print(rmse)