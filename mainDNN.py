"""
#!/usr/bin/env python
# coding: utf-8
# Angle of Arrival Estimation 
# Jianyuan Jet Yu, jianyuan@vt.edu

# flexiable version w/ tensorflow
feature:
    1. early stop
    2. radius based function (RBF)
    3. drop out
    4. batch norm
    5. regularization


"""








import time
import os
import math
import numpy as np
import hdf5storage as h5
import random



# ======================== global setting ======================== 

dataID = './Figs/5_DNN/'   
method = '512x3_p12/'
patience = 12


print(patience)
print(dataID)
print(method)
print( "===========================================")

if not os.path.isdir( dataID+method):
    os.mkdir( dataID+method)
    
    
note = ' test'  
isSend       = 0     
epochs     = 2000    
batch_size = 64   # todo


t = h5.loadmat(dataID + 'Xtest.mat')
Xtest = t['Xtest']
t = h5.loadmat(dataID + 'Ytest.mat')
Ytest = t['Ytest']
t = h5.loadmat(dataID + 'Xtrain.mat')
Xtrain = t['Xtrain']
t = h5.loadmat(dataID + 'Ytrain.mat')
Ytrain = t['Ytrain']

[Ndat,_]= np.shape(Ytrain)
 


os.environ["KMP_DUPLICATE_LIB_OK"] = "1"   # sometime make "1" for Mac 

TEXT    = "low dataset" 
num_bins = 50

tic = time.time()


import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
def get_session(gpu_fraction=1):
    
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Lambda, Reshape, Conv1D, Conv2D,\
        AveragePooling2D,Flatten, Dropout, SimpleRNN, LSTM, concatenate, Layer
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
# from IPython.core.display import Image, display
import matplotlib.pyplot as plt
from numpy import ma
import scipy.io as sio
# from IPython.display import Image
from matplotlib import cm as CM
# from nbconvert import HTMLExporter
import keras
keras.__version__

# Visualize training history
from keras import callbacks
tb = callbacks.TensorBoard(log_dir='./logs', histogram_freq=10, batch_size=32,
                           write_graph=True, write_grads=True, write_images=False,
                           embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
# Early stopping  
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')

from datetime import datetime
import scipy.io as sio

from keras.models import model_from_json

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")


#!nvidia-smi

# Distance Functions
def true_dist(y_true, y_pred):
    return np.sqrt(np.square(np.abs(y_pred[:,0]-y_true[:,0]))+ np.square(np.abs(y_pred[:,1]-y_true[:,1])) )
    # to be amend

def dist(y_true, y_pred):    
     return tf.reduce_mean((tf.sqrt(tf.square(tf.abs(y_pred[:,0]-y_true[:,0]))+ tf.square(tf.abs(y_pred[:,1]-y_true[:,1])))))  

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff,2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


#  =============== 

# Example for Measurement Quality

  
    





with open(dataID+method+"outPut_TF.txt", "a") as text_file:

    text_file.write( "=== " + note + " === \n" )
    text_file.write( "--- caseID %s  begin --- \n" %(dataID))
    text_file.write( "--- local time  " + dt_string + " --- \n" )
    Ytest = Ytest

    Nval = int(Ndat*0.10)

    Xval = Xtrain[:Nval,:]
    Yval = Ytrain[:Nval]
    Xtrain = Xtrain[Nval+1:,:]
    Ytrain = Ytrain[Nval+1:]
    
    # extend dim
    # Xtrain = Xtrain[:,:,np.newaxis]
    # Xtest = Xtest[:,:,np.newaxis]
    # Xval = Xval[:,:,np.newaxis]

    nn_input  = Input((56,66))
                
    nn_output = Flatten()(nn_input)
    # nn_output = LSTM(units=32, dropout_U = 0.001, dropout_W =0.001)(nn_input)
    # nn_output = LSTM(units=32, dropout_U = 0.001, dropout_W = 0.001)(nn_input)
    # nn_output = LSTM(units=32, dropout_U = 0.001, dropout_W = 0.001)(nn_input)
    #nn_output = Dense(512,activation='relu', kernel_regularizer=regularizers.l2(0.001))(nn_output)
#        nn_output = RBFLayer(64, 0.5)(nn_output)
    # nn_output = Dense(32,activation='relu')(nn_output)  
#        nn_output = Dropout(0.1)(nn_output)
    # nn_output = Dense(32,activation='relu')(nn_output)  
    nn_output = Dense(512,activation='relu')(nn_output)
    # nn_output = Dropout(0.01)(nn_output)
    nn_output = Dense(512,activation='relu')(nn_output)
    # nn_output = Dropout(0.01)(nn_output)
    nn_output = Dense(512,activation='relu')(nn_output)

#        nn_output = Dropout(0.1)(nn_output)
#        nn_output = Dropout(0.2)(nn_output)
#        nn_output = Dense(2048,activation='relu')(nn_output) 
#        nn_output = Dropout(0.2)(nn_output)
#        nn_output = Dense(4096,activation='relu')(nn_output)
    # nn_output = Dense(32,activation='relu')(nn_output)
#        nn_output = Dropout(0.1)(nn_output)
#        nn_output = Dropout(0.2)(nn_output)
    nn_output = Dense(2,activation='linear')(nn_output)
    #  directly output 3 para, non classify
    nn = Model(inputs=nn_input,outputs=nn_output)
    
    nn.compile(optimizer='adam', loss='mse',metrics=[dist])
    nn.summary()

    train_hist = nn.fit(x=Xtrain,y=Ytrain,\
                        batch_size = batch_size ,epochs = epochs ,\
                        validation_data=(Xval, Yval), \
                        shuffle=True,\
                        callbacks=[early_stop])
            
    # Evaluate Performance
    Ypredtrain = nn.predict( Xtrain)
    Ypred = nn.predict( Xtest)
    
    Ypredtrain = Ypredtrain
    Ypred = Ypred
    Ytest = Ytest
    Ytrain = Ytrain
    
    
    err = np.mean(abs(Ytest-Ypred),axis=0);
    print( "--- MAE: --- %s" %(err))
    
    text_file.write( " layer [512] \n")
    text_file.write( " test error %s (deg) \n" %(err))


    
    toc =  time.time()
    timeCost = toc - tic
    print( "--- Totally %s seconds ---" %(timeCost))
    text_file.write( " timeCost %s \n" %(timeCost))
    

    # save model to file
    model_json = nn.to_json()
    with open(dataID+"model_TF.json", "w") as json_file:
        json_file.write(model_json)
        nn.save_weights(dataID+"model_TF.h5")
        print("Saved model to disk")
    
    
    
    # Histogramm of errors on test Area
    plt.figure(2)
   
    plt.hist(err,bins=64)
    plt.ylabel('Number of occurence')
    plt.xlabel('Estimate error (deg)')
    plt.grid(True)  
    plt.title('histogram of estimation error')
    plt.savefig(dataID+method+'hist_TF.png')
    
    
    
    plt.figure(5)
    plt.plot(train_hist.history['dist'])
    plt.plot(train_hist.history['val_dist'])
    plt.title('distance')
    plt.ylabel('distance')
    plt.xlabel('epoch')
    plt.grid(True)  
    plt.legend(['train', 'validate'])
    plt.savefig(dataID+method+'hist_dist.png')
    
    plt.figure(6)
    plt.plot(train_hist.history['loss'])
    plt.plot(train_hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid(True)  
    plt.legend(['train', 'validate'])
    plt.savefig(dataID+method+'hist_loss.png')

    
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
        

    
    text_file.write("--- caseID %s  end --- \n" %(dataID))
    text_file.write( "\n")
    text_file.write( "\n")
    text_file.write( "\n")
    text_file.write( "\n")
    
# save down pred data
sio.savemat(dataID+method+'Ypred_TF.mat', {'Ypred':Ypred})
sio.savemat(dataID+method+'Ytest_TF.mat', {'Ytest':Ytest})

