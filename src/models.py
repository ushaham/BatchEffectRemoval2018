'''
Created on Jul 10, 2018

@author: urishaham
Resnet code: https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
VAE code is based on https://github.com/LynnHo/VAE-Tensorflow
WGAN-gp code is based on https://github.com/LynnHo/DCGAN-LSGAN-WGAN-WGAN-GP-Tensorflow/blob/master/models_mnist.py
'''

from functools import partial
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflib as tl



fc = partial(tl.flatten_fully_connected, activation_fn=None)
lrelu = tf.nn.leaky_relu
relu = tf.nn.relu
batch_norm = partial(slim.batch_norm, scale=True, updates_collections=None)

def _resnet_block_v2(input, block_dim, is_training):
    shortcut = input
    input = batch_norm(input, training)
    input = lrelu(input)
    input = fc(input, block_dim)
    input = batch_norm(input, training)
    input = lrelu(input)
    input = fc(input, block_dim)
    return inputs + shortcut
    


def cytof_G_model():
    
    def Enc(input, n_blocks=3, block_dim=20, code_dim=5, reuse=True, is_training=True):
        with tf.variable_scope('Encoder', reuse=reuse):
            y = batch_norm(input, training)
            y = lrelu(y)
            y = fc(y, block_dim)
            for _ in range(n_blocks):
                y = _resnet_block_v2(y)
            c_mu = fc(y, code_dim)
            c_log_sigma_sq = fc(y, code_dim)
            return c_mu, c_log_sigma_sq
    
    def Dec_a(code, output_dim, n_blocks=3, block_dim=20, reuse=True, is_training=True):
        with tf.variable_scope('Decoder_a', reuse=reuse):
            y = batch_norm(code, training)
            y = lrelu(y)
            y = fc(y, block_dim)
            for _ in range(n_blocks):
                y = _resnet_block_v2(y)
                recon = fc(y, output_dim)
                recon = relu(recon)
            return recon
    
    def Dec_b(code, output_dim, n_blocks=3, block_dim=20, reuse=True, is_training=True):
        with tf.variable_scope('Decoder_b:', reuse=reuse):
            y = batch_norm(code, training)
            y = lrelu(y)
            y = fc(y, block_dim)
            for _ in range(n_blocks):
                y = _resnet_block_v2(y)
                recon = fc(y, output_dim)
                recon = relu(recon)
            return recon
    
    return Enc, Dec_a, Dec_b

    def cytof_D_model(input, n_blocks=3, block_dim=20, reuse=True, is_training=True):
        with tf.variable_scope('Discriminator', reuse=reuse):
            y = batch_norm(code, training)
            y = lrelu(y)
            y = fc(y, block_dim)
            for _ in range(n_blocks):
                y = _resnet_block_v2(y)
            logit = fc(y, 1)
            return logit