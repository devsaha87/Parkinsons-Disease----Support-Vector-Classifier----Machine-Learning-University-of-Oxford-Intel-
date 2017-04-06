# -*- coding: utf-8 -*-
"""
Created on Thu Apr 06 17:36:29 2017

@author: Dev
"""

import pandas as pd
import numpy as np


X = pd.read_csv('C:\Users\Dev\Downloads\parkinsons.data')
X.drop('name', axis = 1, inplace = True)
print X.head()
print X.info
print X.describe()
print X.isnull().sum() 
print X.dtypes 

y = X.status
X.drop('status', axis = 1, inplace = True)
print X.columns # 'status' has been dropped from X


# Experiment with Normalizer(), MaxAbsScaler(), MinMaxScaler(), and StandardScaler().
from sklearn import preprocessing

#T = preprocessing.StandardScaler().fit_transform(X)
#T = preprocessing.MinMaxScaler().fit_transform(X)
#T = preprocessing.Normalizer().fit_transform(X)
T = preprocessing.scale(X)
#T = X # No Change

'''
from sklearn.decomposition import PCA
pca = PCA(n_components = 14)
X_pca = pca.fit_transform(T)
'''
# the accuracy levels off at the same value as before from 7 components onwards.

from sklearn.manifold import Isomap

# nested for loops 
best_score = 0
for k in range(2, 6):
    for l in range(4, 7):
        iso = Isomap(n_neighbors = k, n_components = l)
        X_iso = iso.fit_transform(T)

        # train/test split. 30% test group size, with a random_state equal to 7.
        from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_iso, y, test_size = 0.3, random_state = 7)

        #SVC classifier
        from sklearn.svm import SVC
       
        '''
        model = SVC()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print score
        '''

        # a naive, best-parameter searcher by creating a nested for-loops. The outer for-loop should iterate a variable C
        # from 0.05 to 2, using 0.05 unit increments. The inner for-loop should increment a variable gamma from 0.001 to 0.1, using
        # 0.001 unit increments. 

        for i in np.arange(start = 0.05, stop = 2.05, step = 0.05):
            for j in np.arange(start = 0.001, stop = 0.101, step = 0.001):
                model = SVC(C = i, gamma = j)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                if score > best_score:
                    best_score = score
                    best_C = model.C
                    best_gamma = model.gamma
                    best_n_neighbors = iso.n_neighbors
                    best_n_components = iso.n_components
print "The highest score obtained:", best_score
print "C value:", best_C 
print "gamma value:", best_gamma
print "isomap n_neighbors:", best_n_neighbors
print "isomap n_components:", best_n_components