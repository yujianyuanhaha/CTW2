#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:47:47 2020

@author: jianyuan
"""


#from sklearn import datasets
import time
import os
import math
import numpy as np
import hdf5storage as h5
import random

from sklearn.semi_supervised import LabelPropagation, LabelSpreading


dataID = '/home/jianyuan/Codes/1_CTW2/Figs/5_DNN/'   
method = 'spread_Inner/'

print(dataID)
print(method)
print( "===========================================")

if not os.path.isdir( dataID+method):
    os.mkdir( dataID+method)
    


t = h5.loadmat(dataID + 'X.mat')
X = t['X']
t = h5.loadmat(dataID + 'Y.mat')
Y = t['Y']


# reshape
[Ndat,_]= np.shape(Y)
X = np.reshape(X,(Ndat,56*66))


# get X only -> round -> category 0-N
Y = Y[:,0]
Y = 10* np.round(Y*1.0/10)
# min(Y)=-50 now
Y = ((Y-(-50))/10).astype(int)
# 0-64


Ytest = np.copy( Y[0:999] )   # killer trap
# first 1000 blind
Y[0:999] = -1




# begin
label_spread_model = LabelSpreading()


label_spread_model.fit(X, Y)
Ypred = label_spread_model.predict(  X[0:999])

"""
/home/jianyuan/.conda/envs/NewTensorFlow/lib/python3.7/site-packages/sklearn/semi_supervised/label_propagation.py:293: RuntimeWarning: invalid value encountered in true_divide
  self.label_distributions_ /= normalizer
"""

