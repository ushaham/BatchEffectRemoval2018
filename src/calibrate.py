'''
Created on Jul 10, 2018

@author: urishaham
Resnet code: https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
VAE code is based on https://github.com/LynnHo/VAE-Tensorflow

'''

import os.path
import tensorflow as tf

# ==============================================================================
# =                                inputs arguments                            =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=50, help="number of training epochs")
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help="minibatch size")
parser.add_argument('--optimizer', dest='optimizer', type=int, default='Adam')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--code_dim', dest='code_dim', type=int, default=5, help='dimension of code space')
parser.add_argument('--beta', dest='beta', type=float, default=1., help="KL coefficient")
parser.add_argument('--data_path', dest='datas_path', default='/Data', help="path to data folder")
parser.add_argument('--model', dest='model_name', default='cytof_basic')
parser.add_argument('--experiment_name', dest='experiment_name', default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))

args = parser.parse_args()

