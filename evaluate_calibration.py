'''
Created on Jul 6, 2018

@author: urishaham
'''

import os
from sklearn import decomposition
import numpy as np
import argparse
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import keras.backend as K
import scatterHist as sh
import utils



# ==============================================================================
# =                                inputs arguments                            =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--use_test', dest='use_test', type=int, default=True, 
                    help="wether there are separate test data files")
parser.add_argument('--data_path', dest='data_path', default='./Data', help="path to data folder")
parser.add_argument('--data_type', dest='data_type', default='cytof', help="type of data")

   
args = parser.parse_args()
use_test = args.use_test
data_path = args.data_path
data_type = args.data_type


# ==============================================================================
# =                                 load data                                  =
# ==============================================================================

experiment_name = [x[0] for x in os.walk('./output')]
experiment_name = experiment_name[1].split('/')[2]
calibrated_data_dir = './output/%s/calibrated_data' % experiment_name

source_train_data = np.loadtxt(calibrated_data_dir+'/source_train_data.csv', delimiter=',')
target_train_data = np.loadtxt(calibrated_data_dir+'/target_train_data.csv', delimiter=',')
reconstructed_source_train_data = np.loadtxt(calibrated_data_dir+'/reconstructed_source_train_data.csv'
                                         , delimiter=',')
calibrated_source_train_data = np.loadtxt(calibrated_data_dir+'/calibrated_source_train_data.csv'
                                         , delimiter=',')
reconstructed_target_train_data = np.loadtxt(calibrated_data_dir+'/reconstructed_target_train_data.csv'
                                         , delimiter=',')
source_train_code = np.loadtxt(calibrated_data_dir+'/source_train_code.csv'
                                         , delimiter=',')
target_train_code = np.loadtxt(calibrated_data_dir+'/target_train_code.csv'
                                         , delimiter=',')
if use_test:
    source_test_data = np.loadtxt(calibrated_data_dir+'/source_test_data.csv', delimiter=',')
    target_test_data = np.loadtxt(calibrated_data_dir+'/target_test_data.csv', delimiter=',')
    reconstructed_source_test_data = np.loadtxt(calibrated_data_dir+'/reconstructed_source_test_data.csv'
                                             , delimiter=',')
    calibrated_source_test_data = np.loadtxt(calibrated_data_dir+'/calibrated_source_test_data.csv'
                                             , delimiter=',')
    reconstructed_target_test_data = np.loadtxt(calibrated_data_dir+'/reconstructed_target_test_data.csv'
                                             , delimiter=',')
    source_test_code = np.loadtxt(calibrated_data_dir+'/source_test_code.csv'
                                             , delimiter=',')
    target_test_code = np.loadtxt(calibrated_data_dir+'/target_test_code.csv'
                                             , delimiter=',')

input("Data loaded, press Enter to view reconstructions")
plt.close("all")
# ==============================================================================
# =         visualize calibration and reconstruction in PC subspace            =
# ==============================================================================

# compute PCA
pca = decomposition.PCA()
pca.fit(target_train_data)
pc1 = 0
pc2 = 1
axis1 = 'PC'+str(pc1)
axis2 = 'PC'+str(pc2)

source_train_data_pca = pca.transform(source_train_data)
target_train_data_pca = pca.transform(target_train_data)
reconstructed_source_train_data_pca = pca.transform(reconstructed_source_train_data)
calibrated_source_train_data_pca = pca.transform(calibrated_source_train_data)
reconstructed_target_train_data_pca = pca.transform(reconstructed_target_train_data)
if use_test:
    source_test_data_pca = pca.transform(source_test_data)
    target_test_data_pca = pca.transform(target_test_data)
    reconstructed_source_test_data_pca = pca.transform(reconstructed_source_test_data)
    calibrated_source_test_data_pca = pca.transform(calibrated_source_test_data)
    reconstructed_target_test_data_pca = pca.transform(reconstructed_target_test_data)

# plot reconstructions
sh.scatterHist(target_train_data_pca[:,pc1], target_train_data_pca[:,pc2], 
               reconstructed_target_train_data_pca[:,pc1], 
               reconstructed_target_train_data_pca[:,pc2], 
               axis1, axis2, title="target train data reconstruction", 
               name1='true', name2='recon')
  
sh.scatterHist(source_train_data_pca[:,pc1], source_train_data_pca[:,pc2], 
               reconstructed_source_train_data_pca[:,pc1], 
               reconstructed_source_train_data_pca[:,pc2], 
               axis1, axis2, title="source train data reconstruction", 
               name1='true', name2='recon')
if use_test:   
    sh.scatterHist(target_test_data_pca[:,pc1], target_test_data_pca[:,pc2], 
               reconstructed_target_test_data_pca[:,pc1], 
               reconstructed_target_test_data_pca[:,pc2], 
               axis1, axis2, title="target test data reconstruction", 
               name1='true', name2='recon')
   
    sh.scatterHist(source_test_data_pca[:,pc1], source_test_data_pca[:,pc2], 
               reconstructed_source_test_data_pca[:,pc1], 
               reconstructed_source_test_data_pca[:,pc2], 
               axis1, axis2, title="source test data reconstruction", 
               name1='true', name2='recon')

input("Press Enter to view calibration")
plt.close("all")


# plot data before and after calibration

sh.scatterHist(target_train_data_pca[:,pc1], target_train_data_pca[:,pc2], 
               source_train_data_pca[:,pc1], 
               source_train_data_pca[:,pc2], 
               axis1, axis2, title="train data before calibration", 
               name1='target', name2='source')

sh.scatterHist(reconstructed_target_train_data_pca[:,pc1], 
               reconstructed_target_train_data_pca[:,pc2], 
               calibrated_source_train_data_pca[:,pc1], 
               calibrated_source_train_data_pca[:,pc2], 
               axis1, axis2, title="train data after calibration", 
               name1='target', name2='source')
if use_test:
    sh.scatterHist(target_test_data_pca[:,pc1], target_test_data_pca[:,pc2], 
               source_test_data_pca[:,pc1], 
               source_test_data_pca[:,pc2], 
               axis1, axis2, title="test data before calibration", 
               name1='target', name2='source')

    sh.scatterHist(reconstructed_target_test_data_pca[:,pc1], 
               reconstructed_target_test_data_pca[:,pc2], 
               calibrated_source_test_data_pca[:,pc1], 
               calibrated_source_test_data_pca[:,pc2], 
               axis1, axis2, title="test data after calibration", 
               name1='target', name2='source')

input("Press Enter to view per-marker calibration")
plt.close("all")

# ==============================================================================
# =                               per-marker CDF                               =
# ==============================================================================
    
# plot a few markers before and after calibration
marker_names = ['CD45', 'CD19', 'CD127', 'CD4', 
               'CD8a', 'CD20', 'CD25', 'CD278-Beads',
                'TNFa', 'Beads-Tim3', 'CD27', 'CD14', 'CCR7',
                 'CD28', 'CD152', 'FOXP3', 'CD45RO', 'Beads-INFg',
                  'CD223', 'GzB', 'CD3', 'CD274', 'HLADR', 'Beads-PD1',
                   'CD11b']

if use_test:
    target = target_test_data
    source = source_test_data
    rec_target = reconstructed_target_test_data
    cal_source = calibrated_source_test_data
else:
    target = target_train_data
    source = source_train_data
    rec_target = reconstructed_target_train_data
    cal_source = calibrated_source_train_data

for i in range(np.min([3,target.shape[1]])):
    before_t_marker = target[:,i]
    before_s_marker = source[:,i]
    after_t_marker = rec_target[:,i]
    after_s_marker = cal_source[:,i]
    m = np.min([np.min(before_t_marker), np.min(before_s_marker), np.min(after_t_marker), np.min(after_s_marker)])
    M = np.max([np.max(before_t_marker), np.max(before_s_marker), np.max(after_t_marker), np.max(after_s_marker)])
    x = np.linspace(m, M, num=100)
    before_t_ecdf = ECDF(before_t_marker)
    before_s_ecdf = ECDF(before_s_marker)
    after_t_ecdf = ECDF(after_t_marker)
    after_s_ecdf = ECDF(after_s_marker) 
    bt_ecdf = before_t_ecdf(x)
    bs_ecdf = before_s_ecdf(x)
    at_ecdf = after_t_ecdf(x)
    as_ecdf = after_s_ecdf(x)   
    fig = plt.figure()
    a1 = fig.add_subplot(111)
    #a1.plot(bt_ecdf, '-', color = 'black') 
    #a1.plot(bs_ecdf, '--', color = 'black')   
    a1.plot(bt_ecdf, color = 'blue') 
    a1.plot(bs_ecdf, color = 'red') 
    a1.set_xticklabels([])
    plt.legend(['target before cal.', 'source before cal.'], loc=0 ,prop={'size':16})
    plt.title(marker_names[i])
    plt.show(block=False) 
    fig = plt.figure()
    a2 = fig.add_subplot(111)
    #a2.plot(at_ecdf, '-', color = 'black') 
    #a2.plot(as_ecdf, '--', color = 'black') 
    a2.plot(at_ecdf, color = 'blue') 
    a2.plot(as_ecdf, color = 'red') 
    a2.set_xticklabels([])
    plt.legend(['target after cal.', 'source after cal.'], loc=0 ,prop={'size':16})
    plt.title(marker_names[i])
    plt.show(block=False) 
    
input("Press Enter to view correlations")   
plt.close("all")
 
# ==============================================================================
# =                            Correlation matrices                            =
# ==============================================================================

if use_test:
    target = target_test_data
    source = source_test_data
    rec_target = reconstructed_target_test_data
    cal_source = calibrated_source_test_data
else:
    target = target_train_data
    source = source_train_data
    rec_target = reconstructed_target_train_data
    cal_source = calibrated_source_train_data

# compute the correlation matrices of source and target data before and after calibration 
# and plot a histogram of the values of C_diff = C_source-C_target
    
corr_s = np.corrcoef(source, rowvar=0)
corr_t = np.corrcoef(target, rowvar=0)
corr_cal_s = np.corrcoef(cal_source, rowvar=0)
corr_rec_t = np.corrcoef(rec_target, rowvar=0)


B = corr_s - corr_t
A = corr_cal_s - corr_rec_t

NB = np.linalg.norm(B, 'fro')
NA = np.linalg.norm(A, 'fro')

print('norm of diff of correl matrices before calibration: %.2f' % NB)
print('norm of diff of correl matrices after calibration: %.2f' % NA)


Bf = B.flatten()
Af = A.flatten()

f = np.zeros((Bf.shape[0],2))
f[:,0] = Bf
f[:,1] = Af

fig = plt.figure()
plt.hist(f[:,:2], bins = 10, normed=True, histtype='bar')
plt.legend(['before calib.', 'after calib.'], loc=2)
plt.yticks([])
plt.title('magnitude of difference of correlation coefficients')
plt.show(block=False)

input("Press Enter to view MMD analysis") 
plt.close("all")
# ==============================================================================
# =                                      MMD                                   =
# ==============================================================================

if use_test:
    target = target_test_data
    source = source_test_data
    rec_target = reconstructed_target_test_data
    cal_source = calibrated_source_test_data
else:
    target = target_train_data
    source = source_train_data
    rec_target = reconstructed_target_train_data
    cal_source = calibrated_source_train_data

# MMD in input space

mmd_before = np.zeros(3)
mmd_after = np.zeros(3)
mmd_target_target_before = np.zeros(3)
mmd_target_target_after = np.zeros(3)

num_pts=500

for i in range(3):
    source_inds = np.random.randint(low=0, high = source.shape[0], size = num_pts)
    target_inds = np.random.randint(low=0, high = target.shape[0], size = num_pts)
    target_inds1 = np.random.randint(low=0, high = target.shape[0], size = num_pts)
    mmd_before[i] = K.eval(utils.MMD(source,target).cost(K.variable(value=source[source_inds]), 
              K.variable(value=target[target_inds])))
    mmd_after[i] = K.eval(utils.MMD(cal_source,rec_target).cost(K.variable(value=cal_source[source_inds]), 
             K.variable(value=rec_target[target_inds])))
    mmd_target_target_before[i] = K.eval(utils.MMD(target,target).cost(K.variable(value=target[target_inds]), 
                            K.variable(value=target[target_inds1])))
    mmd_target_target_after[i] = K.eval(utils.MMD(rec_target,rec_target).cost(K.variable(value=rec_target[target_inds]), 
                           K.variable(value=rec_target[target_inds1])))


print('MMD before calibration: %.2f pm %.2f'%(np.mean(mmd_before),np.std(mmd_before)))
print('MMD after calibration: %.2f pm %.2f'%(np.mean(mmd_after),np.std(mmd_after)))
print('MMD target-target before calibration: %.2f pm %.2f'%(np.mean(mmd_target_target_before),
                                                            np.std(mmd_target_target_before)))
print('MMD target-target after calibration: %.2f pm %.2f'%(np.mean(mmd_target_target_after),
                                                           np.std(mmd_target_target_after)))

# MMD in code space
mmd_source_target_train = np.zeros(3)
mmd_source_target_test = np.zeros(3)
mmd_source_source_train = np.zeros(3)
mmd_target_target_train = np.zeros(3)

for i in range(3):
    source_train_inds = np.random.randint(low=0, high = source_train_code.shape[0], size = num_pts)
    source_train_inds1 = np.random.randint(low=0, high = source_train_code.shape[0], size = num_pts)
    source_test_inds = np.random.randint(low=0, high = source_test_code.shape[0], size = num_pts)
    target_train_inds = np.random.randint(low=0, high = target_train_code.shape[0], size = num_pts)
    target_train_inds1 = np.random.randint(low=0, high = target_train_code.shape[0], size = num_pts)
    target_test_inds = np.random.randint(low=0, high = target_test_code.shape[0], size = num_pts)
    mmd_source_target_train[i] = K.eval(utils.MMD(source_train_code,target_train_code).cost(K.variable(value=source_train_code[source_train_inds]), 
              K.variable(value=target_train_code[target_train_inds])))
    mmd_source_target_test[i] = K.eval(utils.MMD(source_test_code,target_test_code).cost(K.variable(value=source_test_code[source_test_inds]), 
              K.variable(value=target_test_code[target_test_inds])))
    mmd_source_source_train[i] = K.eval(utils.MMD(source_train_code,source_train_code).cost(K.variable(value=source_train_code[source_train_inds]), 
              K.variable(value=source_train_code[source_train_inds1])))
    mmd_target_target_train[i] = K.eval(utils.MMD(target_train_code,target_train_code).cost(K.variable(value=target_train_code[target_train_inds]), 
              K.variable(value=target_train_code[target_train_inds1])))

print('MMD source-target_train in code space after calibration: %.2f pm %.2f'%(np.mean(mmd_source_target_train),
                                                                         np.std(mmd_source_target_train)))
print('MMD source-target_test in code space after calibration: %.2f pm %.2f'%(np.mean(mmd_source_target_test),
                                                                         np.std(mmd_source_target_test)))
print('MMD source-source_train in code space after calibration: %.2f pm %.2f'%(np.mean(mmd_source_source_train),
                                                                         np.std(mmd_source_source_train)))
print('MMD target-target_train in code space after calibration: %.2f pm %.2f'%(np.mean(mmd_target_target_train),
                                                                         np.std(mmd_target_target_train)))

input("Press Enter to view CD-8 analysis (for cytof data)") 
plt.close("all")    
# ==============================================================================
# =                              CD8 sub-population                            =
# ==============================================================================
if data_type=='cytof':
    if use_test:
        target_labels = target_test_data
        source_labels = source_test_data
        rec_target = reconstructed_target_test_data
        cal_source = calibrated_source_test_data
        source_label_filename = data_path+"/source_test_labels.csv"
        target_label_filename = data_path+"/target_test_labels.csv" 
    
    else:
        target = target_train_data
        source = source_train_data
        rec_target = reconstructed_target_train_data
        cal_source = calibrated_source_train_data
        source_label_filename = data_path+"/source_train_labels.csv"
        target_label_filename = data_path+"/target_train_labels.csv" 
        
    if os.path.isfile(source_label_filename) & os.path.isfile(target_label_filename):    
    
        source_labels = np.loadtxt(source_label_filename, delimiter=',')
        target_labels = np.loadtxt(target_label_filename, delimiter=',')
        
        source_sub_pop = source[source_labels==1]
        target_sub_pop = target[target_labels==1]
        cal_source_sub_pop = cal_source[source_labels==1]
        rec_target_sub_pop = rec_target[target_labels==1]
        
        marker1 = 13 #17 'IFNg'
        marker2 = 19
        
        axis1 = 'CD28'
        axis2 = 'GzB'
        
        # before calibration
        sh.scatterHist(target_sub_pop[:,marker1], target_sub_pop[:,marker2], source_sub_pop[:,marker1], 
                       source_sub_pop[:,marker2], axis1, axis2, title="data in CD28-GzB plane before calibration", 
                       name1='target', name2='source')
        # after calibration 
        sh.scatterHist(rec_target_sub_pop[:,marker1], rec_target_sub_pop[:,marker2], cal_source_sub_pop[:,marker1], 
                       cal_source_sub_pop[:,marker2], axis1, axis2, title="data in CD28-GzB plane after calibration", 
                       name1='target', name2='source')


input("Press Enter to exit")     
plt.close("all")