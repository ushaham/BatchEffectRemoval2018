#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 16:13:33 2018

@author: urishaham
"""

import os
from sklearn import decomposition
import numpy as np
import argparse
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import traceback
from sklearn import metrics

import scatterHist as sh
import utils
import pylib
import tflib as tl




# ==============================================================================
# =                                inputs arguments                            =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', dest='data_path', default='./Data', 
                    help="path to data folder")
parser.add_argument('--data_type', dest='data_type', default='cytof', 
                    help="type of data, either cytof or other")
parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=200, 
                    help="number of training epochs")
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, 
                    help="minibatch size")
parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=.1, 
                    help='learning rate decay rate')
parser.add_argument('--decay_epochs', dest='decay_epochs', type=float, default=50, 
                    help='epochs till lr decay')
parser.add_argument('--lr', dest='lr', type=float, default=1e-3, 
                    help='initial learning rate')
parser.add_argument('--calib', dest='calib', default=None, 
                    help='org data or calibrated data')


   
args = parser.parse_args()
print(args)
data_path = args.data_path
data_type = args.data_type
n_epochs = args.n_epochs
batch_size = args.batch_size
decay_rate = args.decay_rate
decay_epochs = args.decay_epochs
lr = args.lr
calib = args.calib


# ==============================================================================
# =                                 load data                                  =
# ==============================================================================

experiment_name = [x[0] for x in os.walk('./output')]
experiment_name = experiment_name[1].split('/')[2]
calibrated_data_dir = './output/%s/calibrated_data' % experiment_name

source_train_data = np.loadtxt(calibrated_data_dir+'/source_train_data.csv', delimiter=',')
target_train_data = np.loadtxt(calibrated_data_dir+'/target_train_data.csv', delimiter=',')
source_train_labels = np.loadtxt(data_path + '/source_train_labels.csv', delimiter=',')
target_train_labels = np.loadtxt(data_path + '/target_train_labels.csv', delimiter=',')

calibrated_source_train_data = np.loadtxt(calibrated_data_dir+'/calibrated_source_train_data.csv',
                                          delimiter=',')
reconstructed_target_train_data = np.loadtxt(calibrated_data_dir+'/reconstructed_target_train_data.csv', 
                                             delimiter=',')

source_train_code = np.loadtxt(calibrated_data_dir+'/source_train_code.csv',
                                          delimiter=',')
target_train_code = np.loadtxt(calibrated_data_dir+'/target_train_code.csv', 
                                             delimiter=',')

n_source = source_train_data.shape[0]
n_target = target_train_data.shape[0]

# ==============================================================================
# =                               build datasets                               =
# ==============================================================================
    
input_dim = source_train_data.shape[1]
code_dim = source_train_code.shape[1]

if calib == None:
    dim = input_dim
    source_data = source_train_data
    target_data = target_train_data
    
if calib == 'data':
    dim = input_dim
    source_data = calibrated_source_train_data
    target_data = reconstructed_target_train_data
    
if calib == 'code':
    dim = code_dim
    source_data = source_train_code
    target_data = target_train_code
    
    
    

x_data = tf.placeholder(tf.float32, shape=[None, dim])
y_data = tf.placeholder(tf.int64, shape=[None,])

    
train_dataset = utils.make_dataset2((x_data,y_data), 
                                   batch_size=batch_size, 
                                   repeat=True, 
                                   shuffle=True)
test_dataset = utils.make_dataset2((x_data,y_data), 
                                   batch_size=batch_size, 
                                   repeat=False, 
                                   shuffle=False)

iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)

#iterator = dataset.make_initializable_iterator() # create the iterator
x, y_ = iterator.get_next()

train_init_op = iterator.make_initializer(train_dataset)
test_init_op = iterator.make_initializer(test_dataset)

# ==============================================================================
# =                                    graph                                   =
# ==============================================================================

# inputs
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

# classifier
cell_type_classifier = utils.get_models('Cell_type_classifier')
logits, end_points = cell_type_classifier(x, is_training=is_training)

prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(tf.argmax(logits, 1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=logits)

vars = utils.trainable_variables()


# LR decay policy
iters_per_epoch = int(n_target/batch_size)
decay_steps = iters_per_epoch * decay_epochs


with tf.variable_scope('global_step', reuse=tf.AUTO_REUSE):
    global_step = tf.Variable(0, trainable=False)

learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps, 
                                           decay_rate, staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, 
                                         beta1=0.5)
step = optimizer.minimize(loss, var_list=vars, 
                                      global_step=global_step)


# ==============================================================================
# =                             train classifier                               =
# ==============================================================================

# session
sess = tl.session()

# initialize iterator
sess.run(train_init_op, feed_dict={x_data: target_data, y_data: target_train_labels}) 
    

# saver
saver = tf.train.Saver(max_to_keep=1)

# initialization
ckpt_dir = './output/%s/checkpoints' % experiment_name
pylib.mkdir(ckpt_dir)
sess.run(tf.global_variables_initializer())


# train
overall_it = 0
try:
    for ep in range(n_epochs):
        for it in range(iters_per_epoch):
            overall_it += 1        
            # train classifier
            _ = sess.run(step, feed_dict={is_training: True})
            # display
            if (it + 1) % 10 == 0:
                batch_acc = sess.run(accuracy, feed_dict={is_training: False})
                print("Epoch: (%3d/%5d) iteration: (%5d/%5d) lr: %.6f train batch accuracy: %.3f" 
                      % (ep+1,n_epochs, it+1, iters_per_epoch, 
                         sess.run(optimizer._lr), batch_acc))     
                
    save_path = saver.save(sess, '%s/Epoch_%d.ckpt' % (ckpt_dir, ep))
    print('Model is saved in file: %s' % save_path)
           
    # obtain test accuracy
    
    n_test_batches = int(np.ceil(n_source/batch_size))
    # initialize iterator
    src_prediction = []
    sess.run(test_init_op, feed_dict={x_data: source_data, y_data: source_train_labels}) 
        
    for i in range(n_test_batches):
        src_prediction += sess.run(prediction, 
                                   feed_dict={is_training: False}).tolist()

except:
    traceback.print_exc()    
    
finally:
    sess.close()    
    
source_f = metrics.f1_score(source_train_labels, src_prediction, average='macro')
    
print('F-measure on source data: %.4f' % source_f)
# before calibration:  F-measure on source data: 0.9524
# after calibration: F-measure on source data: 0.9367

