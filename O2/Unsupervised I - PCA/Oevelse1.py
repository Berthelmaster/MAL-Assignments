# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 17:35:03 2020

@author: berth
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
print(digits)


plt.matshow(digits.images[0])
plt.matshow(digits.images[1])
plt.matshow(digits.images[2])

