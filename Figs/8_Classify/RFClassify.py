#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:30:45 2020

@author: jianyuan
"""

"""
#!/usr/bin/env python
# coding: utf-8

"""

import time
import os
import math
import numpy as np
import hdf5storage as h5
import random


# ======================== global setting ======================== 

dataID = '/home/jianyuan/Codes/1_CTW2/Figs/5_DNN/'
dataID2 = '/home/jianyuan/Codes/1_CTW2/Figs/5_DNN/Y_re10/'   
method = 'RF_Classify/'

print(dataID)
print(method)
print( "===========================================")

if not os.path.isdir( dataID+method):
    os.mkdir( dataID+method)
    
    
note = ' test'  
isSend       = 0     # send Email
epochs     = 200    # number of learning epochs


t = h5.loadmat(dataID + 'Xtest.mat')
Xtest = t['Xtest']
t = h5.loadmat(dataID2 + 'Ytest.mat')
Ytest = t['Ytest']
t = h5.loadmat(dataID + 'Xtrain.mat')
Xtrain = t['Xtrain']
t = h5.loadmat(dataID2 + 'Ytrain.mat')
Ytrain = t['Ytrain']

[Ndat,_]= np.shape(Ytrain)
[Ntest,_]= np.shape(Ytest)


# reshape Y 2D to 1D
Xtrain = np.reshape(Xtrain,(Ndat,56*66))
Xtest = np.reshape(Xtest,(Ntest,56*66))

 


os.environ["KMP_DUPLICATE_LIB_OK"] = "1"   # sometime make "1" for Mac 

TEXT    = "low dataset" 
num_bins = 50

tic = time.time()


import random

import matplotlib.pyplot as plt
from numpy import ma
import scipy.io as sio
# from IPython.display import Image
from matplotlib import cm as CM
# from nbconvert import HTMLExporter
import keras
keras.__version__


from datetime import datetime
import scipy.io as sio



now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")


#!nvidia-smi



with open(dataID+method+"outPut_TF.txt", "a") as text_file:

    text_file.write( "=== " + note + " === \n" )
    text_file.write( "--- caseID %s  begin --- \n" %(dataID))
    text_file.write( "--- local time  " + dt_string + " --- \n" )

    
    
    from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
    regr = RandomForestClassifier(n_estimators= 100, max_depth=30, random_state=1)
    regr.fit(Xtrain, Ytrain)
        
    Ypred = regr.predict(Xtest)

    
    err = np.mean(abs(Ytest-Ypred),axis=0);
    print( "--- MAE: --- %s" %(err))
    
    text_file.write( " layer [512] \n")
    text_file.write( " test error %s (deg) \n" %(err))

    
    toc =  time.time()
    timeCost = toc - tic
    print( "--- Totally %s seconds ---" %(timeCost))
    text_file.write( " timeCost %s \n" %(timeCost))

    text_file.write("--- caseID %s  end --- \n" %(dataID))
    text_file.write( "\n")
    text_file.write( "\n")
    text_file.write( "\n")
    text_file.write( "\n")



plt.figure(8)
plt.scatter( Ytest[:,0], Ypred[:,0],facecolors='none',edgecolors='b')
plt.scatter( Ytest[:,1], Ypred[:,1],facecolors='none',edgecolors='r')
plt.title('test - est vs ground')
plt.ylabel('est')
plt.xlabel('ground')
plt.grid(True)  
plt.savefig(dataID+method+'scatter_TF_o.png')

plt.figure(9)
plt.scatter( Ytest[:,0], Ypred[:,0],s=1,facecolors='none',edgecolors='b')
plt.scatter( Ytest[:,1], Ypred[:,1],s=1,facecolors='none',edgecolors='r')
plt.title('test - est vs ground')
plt.ylabel('est')
plt.xlabel('ground')
plt.grid(True)  
plt.savefig(dataID+method+'scatter_TF.png')
    
# save down pred data
sio.savemat(dataID+method+'Ypred_RF.mat', {'Ypred':Ypred})

