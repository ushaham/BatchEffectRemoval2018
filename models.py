'''
Created on Jul 6, 2018

@author: urishaham
Resnet code is based on: https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
VAE code, tflib, pylib are based on https://github.com/LynnHo/VAE-Tensorflow
'''

from functools import partial
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflib as tl



fc = partial(tl.flatten_fully_connected, activation_fn=None)
lrelu = tf.nn.leaky_relu
relu = tf.nn.relu
batch_norm = partial(slim.batch_norm, scale=True, updates_collections=None)

def _resnet_block_v2(inputs, block_dim, is_training):
    shortcut = inputs
    inputs = batch_norm(inputs, is_training)
    inputs = lrelu(inputs)
    inputs = fc(inputs, block_dim)
    inputs = batch_norm(inputs, is_training)
    inputs = lrelu(inputs)
    inputs = fc(inputs, block_dim)
    return inputs + shortcut
    


def cytof_basic():
    def Enc(inputs, n_blocks=3, block_dim=20, code_dim=5, reuse=True, is_training=True):
        with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
            y = batch_norm(inputs, is_training)
            y = lrelu(y)
            y = fc(y, block_dim)
            for _ in range(n_blocks):
                y = _resnet_block_v2(y, block_dim, is_training)
            c_mu = fc(y, code_dim)
            c_log_sigma_sq = fc(y, code_dim)
            return c_mu, c_log_sigma_sq
    
    def Dec_a(code, output_dim, n_blocks=3, block_dim=20, reuse=True, is_training=True):
        with tf.variable_scope('Decoder_a', reuse=tf.AUTO_REUSE):
            y = batch_norm(code, is_training)
            y = lrelu(y)
            y = fc(y, block_dim)
            for _ in range(n_blocks):
                y = _resnet_block_v2(y, block_dim, is_training)
                recon = fc(y, output_dim)
                recon = relu(recon)
                return recon
    
    def Dec_b(code, output_dim, n_blocks=3, block_dim=20, reuse=True, is_training=True):
        with tf.variable_scope('Decoder_b', reuse=tf.AUTO_REUSE):
            y = batch_norm(code, is_training)
            y = lrelu(y)
            y = fc(y, block_dim)
            for _ in range(n_blocks):
                y = _resnet_block_v2(y, block_dim, is_training)
                recon = fc(y, output_dim)
                recon = relu(recon)
                return recon
            
    def Disc(code, n_blocks=3, block_dim=20, reuse=True, is_training=True):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            y = batch_norm(code, is_training)
            y = lrelu(y)
            y = fc(y, block_dim)
            for _ in range(n_blocks):
                y = _resnet_block_v2(y, block_dim, is_training)
            output = fc(y, 1)
            return output    
    
    return Enc, Dec_a, Dec_b, Disc

    