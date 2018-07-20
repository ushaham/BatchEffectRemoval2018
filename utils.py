"""
Created on Tue Jul 10 15:02:29 2018

@author: urishaham
"""

import numpy as np
import os.path
import tensorflow as tf
from numpy import *
import sklearn.preprocessing as prep
from sklearn.neighbors import NearestNeighbors
import models
import keras.backend as K



def get_data(path, data_type, use_test):
    source_train_data_filename = path+"/source_train_data.csv"
    target_train_data_filename = path+"/target_train_data.csv"
    source_test_data_filename = path+"/source_test_data.csv"
    target_test_data_filename = path+"/target_test_data.csv"
    
    source_train_data = np.loadtxt(source_train_data_filename, delimiter=',')
    target_train_data = np.loadtxt(target_train_data_filename, delimiter=',')
    min_n = np.min([len(source_train_data), len(target_train_data)])
    source_train_data[isnan(source_train_data)] = 0
    target_train_data[isnan(target_train_data)] = 0
    if use_test & os.path.isfile(source_test_data_filename):
        print('using source test data: '+source_test_data_filename )
        source_test_data = np.loadtxt(source_test_data_filename, delimiter=',')
        source_test_data[isnan(source_test_data)] = 0
    else:    
        source_test_data = source_train_data
        print('using same source  data for training and testing')
    if use_test &  os.path.isfile(target_test_data_filename):  
        print('using target test data: '+target_test_data_filename )
        target_test_data = np.loadtxt(target_test_data_filename, delimiter=',')
        target_test_data[isnan(target_test_data)] = 0
    else:
        target_test_data = target_train_data
        print('using same target  data for training and testing')
    # do log transformation for cytof data    
    if data_type == 'cytof':
        source_train_data = preProcessCytofData(source_train_data)
        source_test_data = preProcessCytofData(source_test_data)
        target_train_data = preProcessCytofData(target_train_data)
        target_test_data = preProcessCytofData(target_test_data)
        
    source_train_data, source_test_data, _ = standard_scale(source_train_data, source_test_data)
    target_train_data, target_test_data, preprocessor = standard_scale(target_train_data, target_test_data)
        
    return  source_train_data, target_train_data, source_test_data, target_test_data, min_n, preprocessor

def recover_org_scale(X, data_type, preprocessor):
    X = preprocessor.inverse_transform(X)
    # invert log transformation for cytof data    
    if data_type == 'cytof':
        X = np.exp(X) - 1
    return X    

def make_dataset(data, batch_size = 100, buffer_size=4096):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(buffer_size)
    #dataset = dataset.batch(batch_size)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size)) # instead of previous row
    dataset = dataset.repeat()
    return dataset

def get_models(model_name):
    return getattr(models, model_name)()

def gradient_penalty(real, fake, f):
        def interpolate(a, b):
            shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

        x = interpolate(real, fake)
        pred = f(x)
        gradients = tf.gradients(pred, x)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=1))
        gp = tf.reduce_mean((slopes - 1.)**2)
        return gp
    
def tensors_filter(tensors, filters, combine_type='or'):
    assert isinstance(tensors, (list, tuple)), '`tensors` shoule be a list or tuple!'
    assert isinstance(filters, (str, list, tuple)), \
        '`filters` should be a string or a list(tuple) of strings!'
    assert combine_type == 'or' or combine_type == 'and', "`combine_type` should be 'or' or 'and'!"

    if isinstance(filters, str):
        filters = [filters]

    f_tens = []
    for ten in tensors:
        if combine_type == 'or':
            for filt in filters:
                if filt in ten.name:
                    f_tens.append(ten)
                    break
        elif combine_type == 'and':
            all_pass = True
            for filt in filters:
                if filt not in ten.name:
                    all_pass = False
                    break
            if all_pass:
                f_tens.append(ten)
    return f_tens    

def trainable_variables(filters=None, combine_type='or'):
    t_var = tf.trainable_variables()
    if filters is None:
        return t_var
    else:
        return tensors_filter(t_var, filters, combine_type)
    
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    if X_test is not None:
        X_test = preprocessor.transform(X_test)
    return X_train, X_test, preprocessor    

def preProcessCytofData(data):
    return np.log(1+data)

def squaredDistance(X,Y):
    # X is nxd, Y is mxd, returns nxm matrix of all pairwise Euclidean distances
    # broadcasted subtraction, a square, and a sum.
    r = K.expand_dims(X, axis=1)
    return K.sum(K.square(r-Y), axis=-1)

 
 
class MMD:
    MMDTargetTrain = None
    MMDTargetTrainSize = None
    MMDTargetValidation = None
    MMDTargetValidationSize = None
    MMDTargetSampleSize = None
    kernel = None
    scales = None
    weights = None
    
    def __init__(self,
                 MMDLayer,
                 MMDTargetTrain,
                 MMDTargetValidation_split=0.1,
                 MMDTargetSampleSize=1000,
                 n_neighbors = 25,
                 scales = None,
                 weights = None):
        if scales == None:
            print("setting scales using KNN")
            med = np.zeros(20)
            for ii in range(1,20):
                sample = MMDTargetTrain[np.random.randint(MMDTargetTrain.shape[0], size=MMDTargetSampleSize),:]
                nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(sample)
                distances,dummy = nbrs.kneighbors(sample)
                #nearest neighbor is the point so we need to exclude it
                med[ii]=np.median(distances[:,1:n_neighbors])
            med = np.median(med)  
            scales = [med/2, med, med*2] # CyTOF    
            #print(scales)
        scales = K.variable(value=np.asarray(scales))
        if weights == None:
            print("setting all scale weights to 1")
            weights = K.eval(K.shape(scales)[0])
        weights = K.variable(value=np.asarray(weights))
        self.MMDLayer =  MMDLayer
        self.MMDTargetTrain = K.variable(value=MMDTargetTrain)
        self.MMDTargetTrainSize = K.eval(K.shape(self.MMDTargetTrain)[0])
        self.MMDTargetSampleSize = MMDTargetSampleSize
        self.kernel = self.RaphyKernel
        self.scales = scales
        self.weights = weights

    def RaphyKernel(self,X,Y):
            #expand dist to a 1xnxm tensor where the 1 is broadcastable
            sQdist = K.expand_dims(squaredDistance(X,Y),0) 
            #expand scales into a px1x1 tensor so we can do an element wise exponential
            self.scales = K.expand_dims(K.expand_dims(self.scales,-1),-1)
            #expand scales into a px1x1 tensor so we can do an element wise exponential
            self.weights = K.expand_dims(K.expand_dims(self.weights,-1),-1)
            #calculated the kernal for each scale weight on the distance matrix and sum them up
            return K.sum(self.weights*K.exp(-sQdist / (K.pow(self.scales,2))),0)
    
    #Calculate the MMD cost
    def cost(self,source, target):
        #calculate the 3 MMD terms
        xx = self.kernel(source, source)
        xy = self.kernel(source, target)
        yy = self.kernel(target, target)
        #calculate the bias MMD estimater (cannot be less than 0)
        MMD = K.mean(xx) - 2 * K.mean(xy) + K.mean(yy)
        #return the square root of the MMD because it optimizes better
        return K.sqrt(MMD);