#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2022/6/29 11:53
# @File  : biplot.py
# @Author: 
# @Desc  : PCA biplot图， PCA结果和属性成分相关圈都绘制在一张图上

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
feature_names = iris.feature_names
X = iris.data
y = iris.target

# In general, it's a good idea to scale the data prior to PCA.
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
pca = PCA()
x_new = pca.fit_transform(X)

def biplot(score,coeff,labels=None):
    """
    :param score:
    :param coeff:
    :param labels: 特征的名字
    """
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

#Call the function. Use only the 2 PCs.
score = x_new[:,0:2]  # [150,2], 降维后的主成分
coeff_component_and_attribute = pca.components_[0:2, :]  # [2,4], 主成分和属性的系数
coeff = np.transpose(coeff_component_and_attribute)  #shape: [4,2], 做下转置， 特征空间中的主轴，代表数据中最大方差的方向。相当于，居中输入数据的右侧奇异向量，与它的特征向量平行。这些成分按解释方差排序
biplot(score,coeff, labels=feature_names)
plt.show()
