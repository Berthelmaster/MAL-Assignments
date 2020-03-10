# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:09:49 2020

@author: Thomas Berthel
"""


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from random import randint

#Øvelse 2

#Start of A

digits = load_digits()
X_digits, y_digits = load_digits(return_X_y=True)

print(digits.keys())
print(digits.data.shape)

X_data = scale(digits.data)

y_data = digits.target

#Max = digits.data.shape[1] = 64
pca = PCA(n_components=64)
X_pca_data = pca.fit(X_data.T)

explainedvariance = X_pca_data.explained_variance_

plt.plot(np.cumsum(explainedvariance) / np.sum(explainedvariance))

#End of A

#Start of B

pcaB = PCA(n_components=36)

newshape = (8, 8)

digits_transform = pcaB.fit_transform(scale(X_digits))

digits_reconstructor = pcaB.inverse_transform(digits_transform)

digit_recon = digits_reconstructor[1,:]
digit_recon = digit_recon.reshape(newshape)
# Found out how to actually print the image on the internet
plt.imshow(digit_recon)
plt.title("After PCA inverse transform, n_components=36")
plt.show()

#Normal image:
plt.matshow(digits.images[1])
plt.title("Pure data from digits.images")

#End of B

#Start of C

#Før kompression (1797,64)
#Efter kompression (1797,36)

#End of C
    
