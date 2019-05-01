# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 21:58:17 2019

@author: VINIT KORADE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("D:\\LP3\\ML\\Exp3\\data.csv")
x = data.iloc[:,:-1].values
y = data.iloc[:,2].values

#KNN
from sklearn.neighbors import KNeighborsClassifier
Knn = KNeighborsClassifier(n_neighbors= 3)
Knn.fit(x, y)

X_pred = np.array([6,6])
Y_pred = Knn.predict(X_pred.reshape([1,-1]))
print("\nClass of [6,6] using KNN:- ", Y_pred[0])

#Weighted KNN
Knn = KNeighborsClassifier(n_neighbors= 3, weights='distance')
Knn.fit(x, y)

Y_pred = Knn.predict(X_pred.reshape([1,-1]))
print("\nClass of [6,6] using Weighted KNN:- ", Y_pred[0])

plt.scatter(x[0][0], x[0][1], c='orange', marker='s')
plt.scatter(x[1][0], x[1][1], c='orange', marker='s')
plt.scatter(x[2][0], x[2][1], c='blue')
plt.scatter(x[3][0], x[3][1], c='orange', marker='s')
plt.scatter(x[4][0], x[4][1], c='blue')
plt.scatter(x[5][0], x[5][1], c='orange', marker='s')

if (Y_pred==0):
    color = 'orange'
    marker = 's'
else:
    color = 'blue'
    marker = '.'
    
plt.scatter(X_pred[0], X_pred[1], c=color, marker=marker, s=300)