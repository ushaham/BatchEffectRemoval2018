#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 09:21:42 2018

@author: urishaham

This scripts produces the TSNE plots appearing in Section 4.2 of the manuscript

"""
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pylib




experiment_name = [x[0] for x in os.walk('./output')]
experiment_name = experiment_name[1].split('/')[2]
calibrated_data_dir = './output/%s/calibrated_data' % experiment_name
plots_dir = './output/%s/tsne_plots' % experiment_name
pylib.mkdir(plots_dir)

# load data
source_train_data = np.loadtxt(calibrated_data_dir+'/source_train_data.csv', delimiter=',')
target_train_data = np.loadtxt(calibrated_data_dir+'/target_train_data.csv', delimiter=',')
calibrated_source_train_data = np.loadtxt(calibrated_data_dir+'/calibrated_source_train_data.csv'
                                         , delimiter=',')
reconstructed_target_train_data = np.loadtxt(calibrated_data_dir+'/reconstructed_target_train_data.csv'
                                         , delimiter=',')
n_s = source_train_data.shape[0]
n_t = target_train_data.shape[0]

before_calib = np.concatenate([source_train_data,target_train_data], axis=0)
after_calib = np.concatenate([calibrated_source_train_data,
                              reconstructed_target_train_data], axis=0)

# embed data
embedding_before = TSNE(n_components=2).fit_transform(before_calib)
embedding_after = TSNE(n_components=2).fit_transform(after_calib)

# visualize
fig = plt.figure(figsize=(8, 8))
plt.scatter(embedding_before[:n_s,0], embedding_before[:n_s,1], color = 'blue', s=3)
plt.scatter(embedding_before[n_s:,0], embedding_before[n_s:,1], color = 'red', s=3)
plt.title('TSNE embedding before calibration', fontsize=15)
plt.legend(['batch 1', 'batch 2'], fontsize=15)
plt.show
fig.savefig(plots_dir+'/tsne_embedding_before_calib.png')

fig = plt.figure(figsize=(8, 8))
plt.scatter(embedding_after[:n_s,0], embedding_after[:n_s,1], color = 'blue', s=3)
plt.scatter(embedding_after[n_s:,0], embedding_after[n_s:,1], color = 'red', s=3)
plt.title('TSNE embedding after calibration', fontsize=15)
plt.legend(['batch 1', 'batch 2'], fontsize=15)
plt.show
fig.savefig(plots_dir+'/tsne_embedding_after_calib.png')

