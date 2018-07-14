'''
Created on Jul 6, 2018

@author: urishaham
'''

import os
from sklearn import decomposition
import numpy as np
import scatterHist as sh





# MMD of same biology and different batch in code space, with and without adv loss
   

# ==============================================================================
# =                                 load data                                  =
# ==============================================================================

experiment_name = [x[0] for x in os.walk('./output')]
experiment_name = experiment_name[1].split('/')[2]
calibrated_data_dir = './output/%s/calibrated_data' % experiment_name

source_test_data = np.loadtxt(calibrated_data_dir+'/source_test_data.csv', delimiter=',')
target_test_data = np.loadtxt(calibrated_data_dir+'/target_test_data.csv', delimiter=',')
reconstructed_source_test_data = np.loadtxt(calibrated_data_dir+'/reconstructed_source_test_data.csv'
                                         , delimiter=',')
calibrated_source_test_data = np.loadtxt(calibrated_data_dir+'/calibrated_source_test_data.csv'
                                         , delimiter=',')
reconstructed_target_test_data = np.loadtxt(calibrated_data_dir+'/reconstructed_target_test_data.csv'
                                         , delimiter=',')

# ==============================================================================
# =         visualize calibration and reconstruction in PC subspace            =
# ==============================================================================

# compute PCA
pca = decomposition.PCA()
pca.fit(target_test_data)
pc1 = 0
pc2 = 1
axis1 = 'PC'+str(pc1)
axis2 = 'PC'+str(pc2)

source_test_data_pca = pca.transform(source_test_data)
target_test_data_pca = pca.transform(target_test_data)
reconstructed_source_test_data_pca = pca.transform(reconstructed_source_test_data)
calibrated_source_test_data = pca.transform(calibrated_source_test_data)
reconstructed_target_test_data_pca = pca.transform(reconstructed_target_test_data)

# plot reconstructions
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


# plot data before and after calibration
sh.scatterHist(target_test_data_pca[:,pc1], target_test_data_pca[:,pc2], 
               source_test_data_pca[:,pc1], 
               source_test_data_pca[:,pc2], 
               axis1, axis2, title="before calibration", 
               name1='target', name2='source')

sh.scatterHist(reconstructed_target_test_data_pca[:,pc1], 
               reconstructed_target_test_data_pca[:,pc2], 
               calibrated_source_test_data[:,pc1], 
               calibrated_source_test_data[:,pc2], 
               axis1, axis2, title="after calibration", 
               name1='target', name2='source')


##################################### qualitative evaluation: per-marker empirical cdfs #####################################
# plot a few markers before and after calibration
markerNames = ['CD45', 'CD19', 'CD127', 'CD4', 
               'CD8a', 'CD20', 'CD25', 'CD278-Beads',
                'TNFa', 'Beads-Tim3', 'CD27', 'CD14', 'CCR7',
                 'CD28', 'CD152', 'FOXP3', 'CD45RO', 'Beads-INFg',
                  'CD223', 'GzB', 'CD3', 'CD274', 'HLADR', 'Beads-PD1',
                   'CD11b']
for i in range(3):#np.min([10,target.shape[1]])):
    targetMarker = target[:,i]
    beforeMarker = source[:,i]
    afterMarker = calibratedSource_resNet[:,i]
    m = np.min([np.min(targetMarker), np.min(beforeMarker), np.min(afterMarker)])
    M = np.max([np.max(targetMarker), np.max(beforeMarker), np.max(afterMarker)])
    x = np.linspace(m, M, num=100)
    target_ecdf = ECDF(targetMarker)
    before_ecdf = ECDF(beforeMarker)
    after_ecdf = ECDF(afterMarker)   
    tgt_ecdf = target_ecdf(x)
    bf_ecdf = before_ecdf(x)
    af_ecdf = after_ecdf(x)    
    fig = plt.figure()
    a1 = fig.add_subplot(111)
    a1.plot(tgt_ecdf, '-', color = 'black') 
    a1.plot(bf_ecdf, '--', color = 'black') 
    a1.plot(af_ecdf, ':', color = 'black') 
    #a1.plot(tgt_ecdf, color = 'blue') 
    #a1.plot(bf_ecdf, color = 'red') 
    #a1.plot(af_ecdf, color = 'green') 
    a1.set_xticklabels([])
    plt.legend(['target', 'before calibration', 'after calibration'], loc=0 ,prop={'size':16})
    plt.title(markerNames[i])
    plt.show() 
       
##################################### Correlation matrices ##############################################
# compute the correlation matrices C_source, C_target before and after calibration 
# and plot a histogram of the values of C_diff = C_source-C_target
corrB = np.corrcoef(source, rowvar=0)
corrA_resNet = np.corrcoef(calibratedSource_resNet, rowvar=0)
corrA_MLP = np.corrcoef(calibratedSource_MLP, rowvar=0)

corrT = np.corrcoef(target, rowvar=0)
FB = corrT - corrB
FA_resNet = corrT - corrA_resNet
FA_MLP= corrT - corrA_MLP

NB = np.linalg.norm(FB, 'fro')
NA_resNet = np.linalg.norm(FA_resNet, 'fro')
NA_MLP = np.linalg.norm(FA_MLP, 'fro')


print('norm before calibration:         ', str(NB))
print('norm after calibration (resNet): ', str(NA_resNet)) 
print('norm after calibration (MLP):    ', str(NA_MLP)) 



fa_resNet = FA_resNet.flatten()
fa_MLP = FA_MLP.flatten()
fb = FB.flatten()

f = np.zeros((fa_resNet.shape[0],3))
f[:,0] = fb
f[:,1] = fa_resNet
f[:,2] = fa_MLP

fig = plt.figure()
plt.hist(f[:,:2], bins = 10, normed=True, histtype='bar')
plt.legend(['before calib.', 'ResNet calib.', 'MLP calib.'], loc=2)
plt.yticks([])
plt.show()

##################################### quantitative evaluation: MMD #####################################

# MMD of input train data (show that diff is significant)
# MMD of calib train data (show diff is insignificant)
# MMD of code train data (show diff is insignificant)
# 



# MMD with the scales used for training 
mmd_before = np.zeros(5)
mmd_after_resNet = np.zeros(5)
mmd_after_MLP = np.zeros(5)
mmd_target_target = np.zeros(5)

for i in range(5):
    sourceInds = np.random.randint(low=0, high = source.shape[0], size = 1000)
    targetInds = np.random.randint(low=0, high = target.shape[0], size = 1000)
    targetInds1 = np.random.randint(low=0, high = target.shape[0], size = 1000)
    mmd_before[i] = K.eval(cf.MMD(source,target).cost(K.variable(value=source[sourceInds]), K.variable(value=target[targetInds])))
    mmd_after_resNet[i] = K.eval(cf.MMD(calibratedSource_resNet,target).cost(K.variable(value=calibratedSource_resNet[sourceInds]), K.variable(value=target[targetInds])))
    mmd_after_MLP[i] = K.eval(cf.MMD(calibratedSource_MLP,target).cost(K.variable(value=calibratedSource_MLP[sourceInds]), K.variable(value=target[targetInds])))
    mmd_target_target[i] = K.eval(cf.MMD(target,target).cost(K.variable(value=target[targetInds]), K.variable(value=target[targetInds1])))


print('MMD before calibration:         ' + str(np.mean(mmd_before))+'pm '+str(np.std(mmd_before)))
print('MMD after calibration (resNet): ' + str(np.mean(mmd_after_resNet))+'pm '+str(np.std(mmd_after_resNet)))
print('MMD after calibration (MLP):    ' + str(np.mean(mmd_after_MLP))+'pm '+str(np.std(mmd_after_MLP)))
print('MMD target-target:              ' + str(np.mean(mmd_target_target))+'pm '+str(np.std(mmd_target_target)))


##################################### CD8 sub-population #####################################
sourceLabels = genfromtxt(sourceLabelPath, delimiter=',', skip_header=0)
targetLabels = genfromtxt(targetLabelPath, delimiter=',', skip_header=0)

source_subPop = source[sourceLabels==1]
resNetCalibSubPop = calibratedSource_resNet[sourceLabels==1]
mlpCalibSubPop = calibratedSource_MLP[sourceLabels==1]
target_subPop = target[targetLabels==1]

marker1 = 13 #17 'IFNg'
marker2 = 19

axis1 = 'CD28'
axis2 = 'GzB'

# before calibration
sh.scatterHist(target_subPop[:,marker1], target_subPop[:,marker2], source_subPop[:,marker1], source_subPop[:,marker2], axis1, axis2)
# after calibration using ResNet
sh.scatterHist(target_subPop[:,marker1], target_subPop[:,marker2], resNetCalibSubPop[:,marker1], resNetCalibSubPop[:,marker2], axis1, axis2)
# after calibration using MLP (no shortcut connections)
sh.scatterHist(target_subPop[:,marker1], target_subPop[:,marker2], mlpCalibSubPop[:,marker1], mlpCalibSubPop[:,marker2], axis1, axis2)


