'''
Created on Jul 6, 2018

@author: urishaham
Resnet code is based on: https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
VAE code, tflib, pylib are based on https://github.com/LynnHo/VAE-Tensorflow
Transformer code is based on https://github.com/Kyubyong/transformer
'''

from functools import partial
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflib as tl



fc = partial(tl.flatten_fully_connected, activation_fn=None)
lrelu = tf.nn.leaky_relu
relu = tf.nn.relu
batch_norm = partial(slim.batch_norm, scale=True, updates_collections=None)

def _resnet_block_v2(inputs, 
                     block_dim, 
                     is_training,
                     reuse=None):
    
    with tf.variable_scope("resnet_block", reuse=reuse):
        shortcut = inputs
        inputs = batch_norm(inputs, is_training)
        inputs = lrelu(inputs)
        inputs = fc(inputs, block_dim)
        inputs = batch_norm(inputs, is_training)
        inputs = lrelu(inputs)
        inputs = fc(inputs, block_dim)
    return inputs + shortcut
    


def resnet():
    
    def Enc(inputs, 
            n_blocks=3, 
            block_dim=20, 
            code_dim=5, 
            is_training=True):
        
        with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
            y = batch_norm(inputs, is_training)
            y = lrelu(y)
            y = fc(y, block_dim)
            for _ in range(n_blocks):
                y = _resnet_block_v2(y, block_dim, is_training)
            c_mu = fc(y, code_dim)
            c_log_sigma_sq = fc(y, code_dim)
        return c_mu, c_log_sigma_sq
    
    def Dec_a(code, 
              output_dim, 
              n_blocks=3, 
              block_dim=20, 
              is_training=True):
        
        with tf.variable_scope('Decoder_a', reuse=tf.AUTO_REUSE):
            y = batch_norm(code, is_training)
            y = lrelu(y)
            y = fc(y, block_dim)
            for _ in range(n_blocks):
                y = _resnet_block_v2(y, block_dim, is_training)
            recon = fc(y, output_dim)
        return recon
    
    def Dec_b(code, 
              output_dim, 
              n_blocks=3, 
              block_dim=20, 
              is_training=True):
        
        with tf.variable_scope('Decoder_b', reuse=tf.AUTO_REUSE):
            y = batch_norm(code, is_training)
            y = lrelu(y)
            y = fc(y, block_dim)
            for _ in range(n_blocks):
                y = _resnet_block_v2(y, block_dim, is_training)
            recon = fc(y, output_dim)
        return recon
            
    def Disc(code, 
             n_blocks=3, 
             block_dim=20, 
             is_training=True):
        
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            y = batch_norm(code, is_training)
            y = lrelu(y)
            y = fc(y, block_dim)
            for _ in range(n_blocks):
                y = _resnet_block_v2(y, block_dim, is_training)
            output = fc(y, 1)
        return output    
    
    return Enc, Dec_a, Dec_b, Disc

def _normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def _multihead_attention(queries, 
                        keys, 
                        is_training,
                        num_units=20, 
                        num_heads=5, 
                        dropout_rate=0,
                        reuse=None):
    '''Applies multihead attention.
    
    Args:
      queries: A 2d tensor with shape of [N, C_q].
      keys: A 2d tensor with shape of [N, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      num_heads: An int. Number of heads.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    
    with tf.variable_scope("multihead_attention", reuse=reuse):
        
        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, C)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=1), axis=0) # (h*N, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=1), axis=0) # (h*N, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=1), axis=0) # (h*N, C/h) 
        
        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_)) # (h*N)
    
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N)
        key_masks = tf.tile(key_masks, [num_heads]) # (h*N)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N)
        
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N)
        
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N)
        query_masks = tf.tile(query_masks, [num_heads]) # (h*N)
        outputs *= query_masks # broadcasting. (N, C)
        
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)
        
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=1) # (N, C)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        outputs = _normalize(outputs) # (N, C)
 
    return outputs


def _feedforward(inputs, 
                 num_units=20,
                reuse=None):
    '''Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, C].
      num_units: an integer, should be same as the same hyperparam in multihead_attention
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope("multihead_attention", reuse=reuse):
        
        
        # Inner layer
        outputs = fc(inputs, num_units)
        outputs = lrelu(outputs)
        
        # Readout layer
        outputs = fc(outputs, num_units)
        outputs = lrelu(outputs)
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = _normalize(outputs)
    
    return outputs

def transformer():
    
    '''
    def Enc(inputs, 
            n_blocks=3, 
            num_units=20, 
            num_heads=5,
            code_dim=5, 
            is_training=True,
            dropout_rate=0):
        
        with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
            y = fc(inputs, num_units)
            y = lrelu(y)
            for _ in range(n_blocks):
                y = _multihead_attention(queries=y, 
                                         keys=y, 
                                         num_units=num_units, 
                                         num_heads=num_heads, 
                                         dropout_rate=dropout_rate,
                                         is_training=is_training)
                y = _feedforward(y,
                                 num_units=num_units))
                
            c_mu = fc(y, code_dim)
            c_log_sigma_sq = fc(y, code_dim)
            return c_mu, c_log_sigma_sq
    
    def Dec_a(code, 
              output_dim, 
              n_blocks=3, 
              num_units=20, 
              num_heads=5,
              is_training=True,
              dropout_rate=0):
        
        with tf.variable_scope('Decoder_a', reuse=tf.AUTO_REUSE):
            y = fc(code, num_units)
            y = lrelu(y)
            for _ in range(n_blocks):
                y = _multihead_attention(queries=y, 
                                         keys=y, 
                                         num_units=num_units, 
                                         num_heads=num_heads, 
                                         dropout_rate=dropout_rate,
                                         is_training=is_training)
                y = _feedforward(y,
                                 num_units=num_units))
                
            recon = fc(y, output_dim)
            return recon
    
    def Dec_b(code, 
              output_dim, 
              n_blocks=3, 
              num_units=20,
              num_heads=5,
              is_training=True,
              dropout_rate=0):
        
        with tf.variable_scope('Decoder_b', reuse=tf.AUTO_REUSE):
            y = fc(code, num_units)
            y = lrelu(y)
            for _ in range(n_blocks):
                y = _multihead_attention(queries=y, 
                                         keys=y, 
                                         num_units=num_units, 
                                         num_heads=num_heads, 
                                         dropout_rate=dropout_rate,
                                         is_training=is_training)
                y = _feedforward(y,
                                 num_units=num_units))
                
            recon = fc(y, output_dim)
            return recon
        
    '''    
    def Enc(inputs, 
            n_blocks=3, 
            block_dim=20, 
            code_dim=5, 
            is_training=True):
        
        with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
            y = batch_norm(inputs, is_training)
            y = lrelu(y)
            y = fc(y, block_dim)
            for _ in range(n_blocks):
                y = _resnet_block_v2(y, block_dim, is_training)
            c_mu = fc(y, code_dim)
            c_log_sigma_sq = fc(y, code_dim)
        return c_mu, c_log_sigma_sq
    
    def Dec_a(code, 
              output_dim, 
              n_blocks=3, 
              block_dim=20, 
              is_training=True):
        
        with tf.variable_scope('Decoder_a', reuse=tf.AUTO_REUSE):
            y = batch_norm(code, is_training)
            y = lrelu(y)
            y = fc(y, block_dim)
            for _ in range(n_blocks):
                y = _resnet_block_v2(y, block_dim, is_training)
            recon = fc(y, output_dim)
        return recon
    
    def Dec_b(code, 
              output_dim, 
              n_blocks=3, 
              block_dim=20, 
              is_training=True):
        
        with tf.variable_scope('Decoder_b', reuse=tf.AUTO_REUSE):
            y = batch_norm(code, is_training)
            y = lrelu(y)
            y = fc(y, block_dim)
            for _ in range(n_blocks):
                y = _resnet_block_v2(y, block_dim, is_training)
            recon = fc(y, output_dim)
        return recon    
            
    def Disc(code, 
             n_blocks=3, 
             num_units=100, 
             num_heads=5,
             is_training=True,
             dropout_rate=0):
        
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            y = fc(code, num_units)
            y = lrelu(y)
            for _ in range(n_blocks):
                y = _multihead_attention(queries=y, 
                                         keys=y, 
                                         num_units=num_units, 
                                         num_heads=num_heads, 
                                         dropout_rate=dropout_rate,
                                         is_training=is_training)
                y = _feedforward(y,
                                 num_units=num_units)
                
            output = fc(y, 1)
            return output    
    
    return Enc, Dec_a, Dec_b, Disc


        
    
    
    
    
    
    