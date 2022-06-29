#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2022/6/29 11:58
# @File  : biplot2.py
# @Author: 
# @Desc  :

from pca import pca
from sklearn import datasets

iris = datasets.load_iris()
feature_names = iris.feature_names
X = iris.data
y = iris.target

# Initialize to reduce the data up to the number of componentes that explains 95% of the variance.
# model = pca(n_components=0.95)

# Or reduce the data towards 2 PCs
model = pca(n_components=2)

# Fit transform
results = model.fit_transform(X)

# Plot explained variance
fig, ax = model.plot()

# Scatter first 2 PCs
fig, ax = model.scatter()

# Make biplot with the number of features
fig, ax = model.biplot(n_feat=4)