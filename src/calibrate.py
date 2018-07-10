'''
Created on Jul 6, 2018

@author: urishaham
The script is based on https://github.com/LynnHo/VAE-Tensorflow
WGAN-gp code is based on on https://github.com/LynnHo/DCGAN-LSGAN-WGAN-WGAN-GP-Tensorflow/blob/master/models_mnist.py

'''
import argparse
import datetime
import json
from functools import partial
import os.path
import tensorflow as tf

import models
import utils
import pylib


# ==============================================================================
# =                                inputs arguments                            =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=50, help="number of training epochs")
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help="minibatch size")
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--code_dim', dest='code_dim', type=int, default=5, help='dimension of code space')
parser.add_argument('--beta', dest='beta', type=float, default=.1, help="KL coefficient")
parser.add_argument('--gamma', dest='gamma', type=float, default=1., help="adversarial loss coefficient")
parser.add_argument('--delta', dest='delta', type=float, default=10., help="gp loss coefficient")
parser.add_argument('--data_path', dest='data_path', default='./Data', help="path to data folder")
parser.add_argument('--data_type', dest='data_type', default='cytof', help="type of data")

parser.add_argument('--model', dest='model_name', default='cytof_basic')
parser.add_argument('--experiment_name', dest='experiment_name', default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))

args = parser.parse_args()

n_epochs = args.n_epochs
batch_size = args.batch_size
lr = args.lr
code_dim = args.code_dim
beta = args.beta
gamma = args.gamma
delta = args.delta
data_path = args.data_path
data_type = args.data_type
model_name = args.model_name
experiment_name = args.experiment_name

pylib.mkdir('./output/%s' % experiment_name)
with open('./output/%s/setting.txt' % experiment_name, 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))
    
# ==============================================================================
# =                            datasets and models                             =
# ==============================================================================

source_train_data, target_train_data, source_test_data, target_test_data = utils.get_data(data_path, data_type)
source_train_dataset = utils.make_dataset(source_train_data)
target_train_dataset = utils.make_dataset(target_train_data)
s_iterator = source_train_dataset.make_initializable_iterator()
s_next_element = s_iterator.get_next()
t_iterator = target_train_dataset.make_initializable_iterator()
t_next_element = t_iterator.get_next()

input_dim = source_train_data.shape[1]
Enc, Dec_a, Dec_b, Disc = utils.get_models(model_name)
Enc = partial(Enc, code_dim=code_dim)
Dec_a = partial(Dec_a, output_dim=input_dim) 
Dec_b = partial(Dec_b, output_dim=input_dim)


# ==============================================================================
# =                                    graph                                   =
# ==============================================================================

def enc_dec(input, is_training=True):
    # encode
    c_mu, c_log_sigma_sq = Enc(input, is_training=is_training)

    # sample a code
    epsilon = tf.random_normal(tf.shape(c_mu))
    if is_training:
        c = c_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * epsilon
    else:
        c = c_mu

    # reconstruct code
    rec_a = Dec_a(c, is_training=is_training)
    rec_b = Dec_b(c, is_training=is_training)

    return c_mu, c_log_sigma_sq, c, rec_a, rec_b

# input
input_a = tf.placeholder(tf.float32, [None, input_dim])
input_b = tf.placeholder(tf.float32, [None, input_dim])

# encode & decode
c_mu_a, c_log_sigma_sq_a, c_a, rec_a, _ = enc_dec(input_a)
c_mu_b, c_log_sigma_sq_b, c_b, _, rec_b = enc_dec(input_b)

# G loss components
rec_loss_a = tf.losses.mean_squared_error(input_a, rec_a)
rec_loss_b = tf.losses.mean_squared_error(input_b, rec_b)
rec_loss = rec_loss_a + rec_loss_b

kld_loss_a = -tf.reduce_mean(0.5 * (1 + c_log_sigma_sq_a - c_mu_a**2 - tf.exp(c_log_sigma_sq_a)))
kld_loss_b = -tf.reduce_mean(0.5 * (1 + c_log_sigma_sq_b - c_mu_b**2 - tf.exp(c_log_sigma_sq_a)))
kld_loss = kld_loss_a + kld_loss_b

adv_loss =  tf.losses.mean_squared_error(ca,cb)

G_loss = rec_loss + kld_loss * beta + adv_loss * gamma

# D loss components
Disc_a = Disc(c_a)
Disc_b = Disc(c_b)
wd_loss = tf.reduce_mean(Disc_a) - tf.reduce_mean(Disc_b)
gp_loss = utils.gradient_penalty(c_a, c_b, Disc)


D_loss = wd_loss + gp_loss * delta


# otpimizers
d_vars = utils.trainable_variables('discriminator')
g_vars = utils.trainable_variables('G')

G_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(G_loss, var_list=g_var)
D_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(D_loss, var_list=d_var)

# summary
G_summary = tl.summary({rec_loss_a: 'rec_loss_a',
                      rec_loss_b: 'rec_loss_b',
                      rec_loss: 'rec_loss',
                      kld_loss_a: 'kld_loss_a',
                      kld_loss_b: 'kld_loss_b',
                      kld_loss: 'kld_loss',
                      adv_loss: 'adv_loss',
                      G_loss: 'G_loss'})
D_summary = tl.summary({wd_loss: 'wd_loss',
                      gp_loss: 'gp_loss',
                      D_loss: 'D_loss'})


# ==============================================================================
# =                                    train                                   =
# ==============================================================================

# session
sess = tl.session()

# saver
saver = tf.train.Saver(max_to_keep=1)

# summary writer
summary_writer = tf.summary.FileWriter('./output/%s/summaries' % experiment_name, sess.graph)

# initialization
ckpt_dir = './output/%s/checkpoints' % experiment_name
pylib.mkdir(ckpt_dir)
try:
    tl.load_checkpoint(ckpt_dir, sess)
except:
    sess.run(tf.global_variables_initializer())

# train
try:

    it = -1
    for ep in range(n_epochs):
        dataset.reset()
        it_per_epoch = it_in_epoch if it != -1 else -1
        it_in_epoch = 0
        for batch in dataset:
            it += 1
            it_in_epoch += 1

            # batch data
            batch_a, batch_b = batch

            # train D
            D_summary_opt, _ = sess.run([D_summary, D_step], feed_dict={input_a: batch_a, input_b:batch_b})
            summary_writer.add_summary(D_summary_opt, it)
            
            # train G
            g_summary_opt, _ = sess.run([G_summary, G_step], feed_dict={input_a: batch_a, input_b:batch_b})
            summary_writer.add_summary(g_summary_opt, it)

            # display
            if (it + 1) % 1 == 0:
                print("Epoch: (%3d) (%5d/%5d)" % (ep, it_in_epoch, it_per_epoch))
                # plot val datawith scatter Hist
            
            if (it + 1) % 1000 == 0:
                save_dir = './output/%s/sample_training' % experiment_name
        save_path = saver.save(sess, '%s/Epoch_%d.ckpt' % (ckpt_dir, ep))
        print('Model is saved in file: %s' % save_path)
except:
    traceback.print_exc()
finally:
    sess.close()



# ==============================================================================
# =                 visualize calibration on test data                         =
# ==============================================================================
pca = decomposition.PCA()
pca.fit(target)

# project data onto PCs
target_sample_pca = pca.transform(target)
projection_before = pca.transform(source)
projection_after = pca.transform(calibratedSource)

# choose PCs to plot
pc1 = 0
pc2 = 1
axis1 = 'PC'+str(pc1)
axis2 = 'PC'+str(pc2)
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_before[:,pc1], projection_before[:,pc2], axis1, axis2)
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_after[:,pc1], projection_after[:,pc2], axis1, axis2)



# ==============================================================================
# =                                  save data                                 =
# ==============================================================================