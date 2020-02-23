# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# quiver plot

import hdf5storage as h5
import numpy as np
import math

Nsig = 2

t = h5.loadmat('/home/jianyuan/Codes/1_CTW2/Figs/5_DNN/512x3_p8/Ypred_TF.mat')
Ypred = t['Ypred']

t = h5.loadmat('/home/jianyuan/Codes/1_CTW2/Figs/5_DNN/Ytest.mat')
Ytest= t['Ytest']



import matplotlib.pyplot as plt
plt.figure(4)
 

error_vectors = Ytest - Ypred

errors = np.sqrt(  error_vectors[:,0]**2 + error_vectors[:,1]**2 ) 

   
   
plt.quiver(Ytest[:,0],Ytest[:,1],
           error_vectors[:,0],error_vectors[:,1],
           errors)
plt.xlabel("x  (meter)")
plt.ylabel("y  (meter)")
plt.grid(True) 
    
plt.title('Quiver plot of ground truth and estimated postions') 
plt.savefig('/home/jianyuan/Codes/1_CTW2/Figs/6_Quiver.png')
plt.savefig('/home/jianyuan/Codes/1_CTW2/Figs/6_Quiver.pdf')


