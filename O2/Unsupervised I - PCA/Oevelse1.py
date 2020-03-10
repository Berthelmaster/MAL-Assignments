# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 17:35:03 2020

@author: Thomas Berthel
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from random import randint

# Øvelse 1

#Start of A

digits = load_digits()

print(digits.keys())

# Plot numbers of dataset

plt.gray()

for number in range(10):
    plt.matshow(digits.images[number])

#END OF A

#START OF B

# Scandizes dataset along any axis
X_data = scale(digits.data)

y_data = digits.target

pca = PCA(n_components=2)
X_pca_data = pca.fit(X_data.T)

# From https://stackoverflow.com/questions/7827530/array-of-colors-in-python
colors = []
for i in range(10):
    colors.append('#%06X' % randint(0, 0xFFFFFF))

fig, ax = plt.subplots()

for number in range(10):
    Arr = np.where(y_data == number)
    x = X_pca_data.components_[0,Arr[0]]
    y = X_pca_data.components_[1,Arr[0]]
    ax.scatter(x, y, c=colors[number], label=number, alpha=0.4)

ax.legend()
ax.grid(True)
ax.set_title('PCA Handwritten digits 0-9')
plt.show()

#PCA dimensionsreduktion


#END OF B