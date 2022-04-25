from smdr import *
import pygfl, sys, math, csv, os
import matplotlib.pylab as plt
import numpy as np
import scipy.stats as st
import random
from random import sample
from collections import defaultdict
from statsmodels.stats.multitest import local_fdr
import nibabel as nib
import time


### Data Generating Function
def data_gen(prior, signal_weights, signal_dist):
    samples = np.zeros((128, 128))
    flips = np.random.random(size=signal_weights.shape) < signal_weights
    for i in range(128):
        for j in range(128):
            if flips[i, j]: 
                samples[i, j] = sample(signal_dist, 1) + np.random.normal(loc=0, scale=1, size=1)
            else:
                samples[i, j] = np.random.normal(loc=0, scale=1, size=1)
    signals = np.zeros((128, 128))
    signals[np.logical_and(prior == 1, flips==1)] = 1
    return {'z': samples,
			'signals': signals,
            'flips': flips}


def data_gen2(prior, signal_weights, signal_dist1, signal_dist2, overlap):
    samples = np.zeros((128, 128))
    flips = np.random.random(size=signal_weights.shape) < signal_weights
    for i in range(128):
        for j in range(128):
            if flips[i, j] and (i, j) not in overlap: 
                samples[i, j] = sample(signal_dist1, 1) + np.random.normal(loc=0, scale=1, size=1)
            elif flips[i, j] and (i, j) in overlap: 
                samples[i, j] = sample(signal_dist2, 1) + np.random.normal(loc=0, scale=1, size=1)
            else:
                samples[i, j] = np.random.normal(loc=0, scale=1, size=1)
    signals = np.zeros((128, 128))
    signals[np.logical_and(prior == 1, flips==1)] = 1
    return {'z': samples,
			'signals': signals,
            'flips': flips}
            
### Simulation Setting
signal_prior = np.zeros((128, 128))
overlap = []
for i in range(128):
    for j in range(128):
        if np.sqrt((i-55)**2+(j-55)**2)<20:
            signal_prior[i, j] = 1
        if np.sqrt((i-70)**2+(j-70)**2)<15:
            signal_prior[i, j] = 1
        if np.sqrt((i-55)**2+(j-55)**2)<20 and np.sqrt((i-70)**2+(j-70)**2)<15:
            overlap.append((i, j))

# generate signal weight map
## saturated+pure
signal_weights1 = signal_prior.copy()
## mixed+pure (deprecated)
#signal_weights2 = 0.5 * signal_prior.copy()
## saturated+noisy
signal_weights3 = signal_prior.copy()
signal_weights3[np.where(signal_weights3==0)] = 0.01
## mixed+noisy (deprecated)
#signal_weights4 = 0.5 * signal_prior.copy()
#signal_weights4[np.where(signal_weights4==0)] = 0.01

# generate two signal distribution 
## poorly seperated
np.random.seed(2333)
signal_dist_poorly5 = list(np.random.normal(loc=0, scale=5, size=128*128))
np.random.seed(2333)
signal_dist_poorly3 = list(np.random.normal(loc=0, scale=3, size=128*128))
np.random.seed(2333)
signal_dist_poorly2 = list(np.random.normal(loc=0, scale=2, size=128*128))
np.random.seed(2333)
signal_dist_poorly1 = list(np.random.normal(loc=0, scale=1, size=128*128))
## well-seperated
np.random.seed(2333)
signal_dist_well = np.zeros((128*128))
for i in range(128*128):
    flip_well = np.random.random(size=1) > 0.5
    if flip_well: signal_dist_well[i] = np.random.normal(loc=-1, scale=1, size=1)
    else: signal_dist_well[i] = np.random.normal(loc=1, scale=1, size=1)
signal_dist_well = list(signal_dist_well)


np.random.seed(2333)
random.seed(2333)
data = data_gen(signal_prior, signal_weights1, signal_dist_poorly2)
z = data['z']
signals = data['flips']


def fnp_cal(de, truth):
    fnp = np.logical_and(truth == 1, de == 0).sum() / float((truth == 1).sum())
    return fnp

def fdr_cal(de, truth):
    if (de == 1).sum() == 0:
        fdr = 0
    else:
        fdr = np.logical_and(truth == 0, de == 1).sum() / float((de == 1).sum())
    return fdr

def fm_cal(de, truth):
    if (de == 1).sum() == 0:
        return 0
    else:
        tp = np.logical_and(truth == 1, de == 1).sum()
        fp = np.logical_and(truth == 0, de == 1).sum()
        fn = np.logical_and(truth == 1, de == 0).sum()
        fm = math.sqrt((tp/(tp+fp))*(tp/(tp+fn)))
        return fm

def f1_cal(de, signals):
    tp = np.logical_and(signals == 1, de == 1).sum()
    fp = np.logical_and(signals == 0, de == 1).sum()
    fn = np.logical_and(signals == 1, de == 0).sum()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall/(precision+recall)
    return f1


def simulation_run(z, signals, fdr_level, epsilon):
    de_bh = bh(z, fdr_level)
    de_fdrl = fdrl(z, fdr_level)
    results_fdrs = fdrs(z, fdr_level)
    de_fdrs = results_fdrs['de']
    lfdr_fdrs = results_fdrs['lfdr']
    
    de_mdr = mdr(z, epsilon)
    de_afnc = afnc(z, epsilon)
    de_smdr005 = smdr(z, lfdr_fdrs, 0.05)
    de_smdr01 = smdr(z, lfdr_fdrs, 0.1)
    de_smdr02 = smdr(z, lfdr_fdrs, 0.2)
    
    signal_est = round(epsest_func(z, 0, 1)*z.size)
    signal_est_prior = round(results_fdrs['c'].sum())
    signal_est_post = round(results_fdrs['post'].sum())
    
    fnp = np.zeros(9)
    fdr = np.zeros(9)
    fm = np.zeros(9)
    f1 = np.zeros(9)
    
    fnp[0] = fnp_cal(de_bh, signals)
    fnp[1] = fnp_cal(de_fdrl, signals)
    fnp[2] = fnp_cal(de_fdrs, signals)
    fnp[3] = fnp_cal(de_mdr, signals)
    fnp[4] = fnp_cal(de_afnc, signals)
    fnp[5] = fnp_cal(de_smdr005, signals)
    fnp[6] = fnp_cal(de_smdr01, signals)
    fnp[7] = fnp_cal(de_smdr02, signals)

    fdr[0] = fdr_cal(de_bh, signals)
    fdr[1] = fdr_cal(de_fdrl, signals)
    fdr[2] = fdr_cal(de_fdrs, signals)
    fdr[3] = fdr_cal(de_mdr, signals)
    fdr[4] = fdr_cal(de_afnc, signals)
    fdr[5] = fdr_cal(de_smdr005, signals)
    fdr[6] = fdr_cal(de_smdr01, signals)
    fdr[7] = fdr_cal(de_smdr02, signals)

    fm[0] = fm_cal(de_bh, signals)
    fm[1] = fm_cal(de_fdrl, signals)
    fm[2] = fm_cal(de_fdrs, signals)
    fm[3] = fm_cal(de_mdr, signals)
    fm[4] = fm_cal(de_afnc, signals)
    fm[5] = fm_cal(de_smdr005, signals)
    fm[6] = fm_cal(de_smdr01, signals)
    fm[7] = fm_cal(de_smdr02, signals)

    f1[0] = f1_cal(de_bh, signals)
    f1[1] = f1_cal(de_fdrl, signals)
    f1[2] = f1_cal(de_fdrs, signals)
    f1[3] = f1_cal(de_mdr, signals)
    f1[4] = f1_cal(de_afnc, signals)
    f1[5] = f1_cal(de_smdr005, signals)
    f1[6] = f1_cal(de_smdr01, signals)
    f1[7] = f1_cal(de_smdr02, signals)

    return {'fnp': fnp,
			'fdr': fdr,
			'fm': fm,
            'f1': f1,
            'de_bh': de_bh,
            'de_fdrl': de_fdrl,
            'de_fdrs': de_fdrs,
            'de_mdr': de_mdr,
            'de_afnc': de_afnc,
            'de_smdr005': de_smdr005,
            'dde_smdr01': de_smdr01,
            'de_smdr02': de_smdr02,
            'signal_est': signal_est,
            'signal_est_prior': signal_est_prior,
            'signal_est_post': signal_est_post}



np.random.seed(2333)
random.seed(2333)
data = data_gen2(signal_prior, signal_weights1, signal_dist_poorly1, signal_dist_poorly3, overlap)
z = data['z']
signals = data['flips']

results = simulation_run(z, signals, 0.05, 0.1)

plt.rcParams["figure.figsize"] = (20,10)
fig, ax = plt.subplots(2,4)

ax[0,0].imshow(signal_prior, cmap='gray_r', vmin=0, vmax=1)
ax[0,0].set_title('Signal region', fontsize=20)
ax[0,0].tick_params(axis='both', which='major', labelsize=14)

ax[0,1].imshow(results['de_bh'], cmap='gray_r', vmin=0, vmax=1)
ax[0,1].set_title('BH-FDR', fontsize=20)
ax[0,1].tick_params(axis='both', which='major', labelsize=14)

ax[0,2].imshow(results['de_fdrl'], cmap='gray_r', vmin=0, vmax=1)
ax[0,2].set_title('FDRL', fontsize=20)
ax[0,2].tick_params(axis='both', which='major', labelsize=14)

ax[0,3].imshow(results['de_fdrs'], cmap='gray_r', vmin=0, vmax=1)
ax[0,3].set_title('FDRS', fontsize=20)
ax[0,3].tick_params(axis='both', which='major', labelsize=14)

ax[1,0].imshow(results['de_afnc'], cmap='gray_r', vmin=0, vmax=1)
ax[1,0].set_title('AFNC', fontsize=20)
ax[1,0].tick_params(axis='both', which='major', labelsize=14)

ax[1,1].imshow(results['de_mdr'], cmap='gray_r', vmin=0, vmax=1)
ax[1,1].set_title('MDR', fontsize=20)
ax[1,1].tick_params(axis='both', which='major', labelsize=14)

ax[1,2].imshow(results['de_new01'], cmap='gray_r', vmin=0, vmax=1)
ax[1,2].set_title('SMDR(beta=0.1)', fontsize=20)
ax[1,2].tick_params(axis='both', which='major', labelsize=14)

ax[1,3].imshow(results['de_new005'], cmap='gray_r', vmin=0, vmax=1)
ax[1,3].set_title('SMDR(beta=0.05)', fontsize=20)
ax[1,3].tick_params(axis='both', which='major', labelsize=14)



np.random.seed(2333)
random.seed(2333)
data = data_gen(signal_prior, signal_weights1, signal_dist_poorly3)
z = data['z']
signals = data['flips']

results = simulation_run(z, signals, 0.05, 0.1)

plt.rcParams["figure.figsize"] = (20,10)
fig, ax = plt.subplots(2,4)

ax[0,0].imshow(z, cmap='gray_r', vmin=0, vmax=1)
ax[0,0].set_title('Signal region', fontsize=20)
ax[0,0].tick_params(axis='both', which='major', labelsize=14)

ax[0,1].imshow(results['de_bh'], cmap='gray_r', vmin=0, vmax=1)
ax[0,1].set_title('BH-FDR', fontsize=20)
ax[0,1].tick_params(axis='both', which='major', labelsize=14)

ax[0,2].imshow(results['de_fdrl'], cmap='gray_r', vmin=0, vmax=1)
ax[0,2].set_title('FDRL', fontsize=20)
ax[0,2].tick_params(axis='both', which='major', labelsize=14)

ax[0,3].imshow(results['de_fdrs'], cmap='gray_r', vmin=0, vmax=1)
ax[0,3].set_title('FDRS', fontsize=20)
ax[0,3].tick_params(axis='both', which='major', labelsize=14)

ax[1,0].imshow(results['de_afnc'], cmap='gray_r', vmin=0, vmax=1)
ax[1,0].set_title('AFNC', fontsize=20)
ax[1,0].tick_params(axis='both', which='major', labelsize=14)

ax[1,1].imshow(results['de_mdr'], cmap='gray_r', vmin=0, vmax=1)
ax[1,1].set_title('MDR', fontsize=20)
ax[1,1].tick_params(axis='both', which='major', labelsize=14)

ax[1,2].imshow(results['de_new01'], cmap='gray_r', vmin=0, vmax=1)
ax[1,2].set_title('SMDR(beta=0.1)', fontsize=20)
ax[1,2].tick_params(axis='both', which='major', labelsize=14)

ax[1,3].imshow(results['de_new005'], cmap='gray_r', vmin=0, vmax=1)
ax[1,3].set_title('SMDR(beta=0.05)', fontsize=20)
ax[1,3].tick_params(axis='both', which='major', labelsize=14)



np.random.seed(2333)
random.seed(2333)
data = data_gen(signal_prior, signal_weights1, signal_dist_poorly1)
z = data['z']
signals = data['flips']

results = simulation_run(z, signals, 0.05, 0.1)

plt.rcParams["figure.figsize"] = (20,10)
fig, ax = plt.subplots(2,4)

ax[0,0].imshow(signal_prior, cmap='gray_r', vmin=0, vmax=1)
ax[0,0].set_title('Signal region', fontsize=20)
ax[0,0].tick_params(axis='both', which='major', labelsize=14)

ax[0,1].imshow(results['de_bh'], cmap='gray_r', vmin=0, vmax=1)
ax[0,1].set_title('BH-FDR', fontsize=20)
ax[0,1].tick_params(axis='both', which='major', labelsize=14)

ax[0,2].imshow(results['de_fdrl'], cmap='gray_r', vmin=0, vmax=1)
ax[0,2].set_title('FDRL', fontsize=20)
ax[0,2].tick_params(axis='both', which='major', labelsize=14)

ax[0,3].imshow(results['de_fdrs'], cmap='gray_r', vmin=0, vmax=1)
ax[0,3].set_title('FDRS', fontsize=20)
ax[0,3].tick_params(axis='both', which='major', labelsize=14)

ax[1,0].imshow(results['de_afnc'], cmap='gray_r', vmin=0, vmax=1)
ax[1,0].set_title('AFNC', fontsize=20)
ax[1,0].tick_params(axis='both', which='major', labelsize=14)

ax[1,1].imshow(results['de_mdr'], cmap='gray_r', vmin=0, vmax=1)
ax[1,1].set_title('MDR', fontsize=20)
ax[1,1].tick_params(axis='both', which='major', labelsize=14)

ax[1,2].imshow(results['de_new01'], cmap='gray_r', vmin=0, vmax=1)
ax[1,2].set_title('SMDR(beta=0.1)', fontsize=20)
ax[1,2].tick_params(axis='both', which='major', labelsize=14)

ax[1,3].imshow(results['de_new005'], cmap='gray_r', vmin=0, vmax=1)
ax[1,3].set_title('SMDR(beta=0.05)', fontsize=20)
ax[1,3].tick_params(axis='both', which='major', labelsize=14)
