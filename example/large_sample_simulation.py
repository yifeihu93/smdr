from smdr import *
import pygfl, sys, math, csv, os
import numpy as np
import scipy.stats as st
import random
from random import sample
from collections import defaultdict
from statsmodels.stats.multitest import local_fdr
import nibabel as nib
import time


########################
## Simulation Setting ##
########################
signal_prior = np.zeros((128, 128))
for i in range(128):
    for j in range(128):
        if np.sqrt((i-55)**2+(j-55)**2)<20:
            signal_prior[i, j] = 1
        if np.sqrt((i-70)**2+(j-70)**2)<15:
            signal_prior[i, j] = 1

# generate signal weight map
## saturated+pure
signal_weights1 = signal_prior.copy()
## mixed+pure
signal_weights2 = 0.5 * signal_prior.copy()
## saturated+noisy
signal_weights3 = signal_prior.copy()
signal_weights3[np.where(signal_weights3==0)] = 0.01
## mixed+noisy
signal_weights4 = 0.5 * signal_prior.copy()
signal_weights4[np.where(signal_weights4==0)] = 0.01
# generate two signal distribution 
## poorly seperated
np.random.seed(2333)
signal_dist_poorly3 = list(np.random.normal(loc=0, scale=3, size=128*128))
np.random.seed(2333)
signal_dist_poorly2 = list(np.random.normal(loc=0, scale=2, size=128*128))
np.random.seed(2333)
signal_dist_poorly2d5 = list(np.random.normal(loc=0, scale=2.5, size=128*128))
## well-seperated
np.random.seed(2333)
signal_dist_well = np.zeros((128*128))
for i in range(128*128):
    flip_well = np.random.random(size=1) > 0.5
    if flip_well: signal_dist_well[i] = np.random.normal(loc=-2, scale=1, size=1)
    else: signal_dist_well[i] = np.random.normal(loc=2, scale=1, size=1)
signal_dist_well = list(signal_dist_well)


np.random.seed(2333)
random.seed(2333)
data = data_gen(signal_prior, signal_weights1, signal_dist_poorly3)
z = data['z']
signals = data['flips']



def fnp_cal(de, truth):
    fnp = np.logical_and(truth == 1, de == 0).sum() / float((truth == 1).sum())
    return fnp

def fdr_cal(de, truth):
    fdr = np.logical_and(truth == 0, de == 1).sum() / float((de == 1).sum())
    return fdr

def fm_cal(de, truth):
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


# ## Simulation Functions
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



# # Simulation Loop

# ### Poorly seperated N(0, 3) with Pure background
#poorly_seperated+pure
np.random.seed(2333)
random.seed(2333)
large_data1 = np.zeros((128, 128, 100))
large_signals1 = np.zeros((128, 128, 100))
for i in range(100):
    large_result1 = data_gen(signal_prior, signal_weights1, signal_dist_poorly3)
    large_data1[0:128, 0:128, i] = large_result1['z']
    large_signals1[0:128, 0:128, i] = large_result1['flips']
    
for i in range(0, 100):
    print(i)
    lfdr = local_fdr(large_data1[0:128, 0:128, i].flatten(), deg=10, nbins=30)


fnp1 = np.zeros((100,7))
fdr1 = np.zeros((100,7))
fm1 = np.zeros((100,7))
signal_est1 = np.zeros(100)
signal_est_post1 = np.zeros(100)
for i in range(0, 100):
    results_1 = simulation_run(large_data1[0:128, 0:128, i], large_signals1[0:128, 0:128, i], 0.05, 0.1)
    fnp1[i,:] = results_1['fnp']
    fdr1[i,:] = results_1['fdr']
    fm1[i,:] = results_1['fm']
    signal_est1[i] = results_1['signal_est']
    signal_est_post1[i] = results_1['signal_est_post']

np.savetxt('fnp1.csv', fnp1, delimiter=',')
np.savetxt('fdr1.csv', fdr1, delimiter=',')
np.savetxt('fm1.csv', fm1, delimiter=',')
np.savetxt('signal_est1.csv', signal_est1, delimiter=',')
np.savetxt('signal_est_post1.csv', signal_est_post1, delimiter=',')


# ### Poorly seperated N(0, 3) with Mixed background
# poorly_seperated+mixed
np.random.seed(2333)
random.seed(2333)
large_data2 = np.zeros((128, 128, 100))
large_signals2 = np.zeros((128, 128, 100))
for i in range(100):
    large_result2 = data_gen(signal_prior, signal_weights3, signal_dist_poorly3)
    large_data2[0:128, 0:128, i] = large_result2['z']
    large_signals2[0:128, 0:128, i] = large_result2['flips']

for i in range(0, 100):
    print(i)
    lfdr = local_fdr(large_data2[0:128, 0:128, i].flatten(), deg=10, nbins=30)

fnp2 = np.zeros((100,7))
fdr2 = np.zeros((100,7))
fm2 = np.zeros((100,7))
signal_est2 = np.zeros(100)
signal_est_post2 = np.zeros(100)
for i in range(0, 100):
    results_2 = simulation_run(large_data2[0:128, 0:128, i], large_signals2[0:128, 0:128, i], 0.05, 0.1)
    fnp2[i,:] = results_2['fnp']
    fdr2[i,:] = results_2['fdr']
    fm2[i,:] = results_2['fm']
    signal_est2[i] = results_2['signal_est']
    signal_est_post2[i] = results_2['signal_est_post']


np.savetxt('fnp2.csv', fnp2, delimiter=',')
np.savetxt('fdr2.csv', fdr2, delimiter=',')
np.savetxt('fm2.csv', fm2, delimiter=',')
np.savetxt('signal_est2.csv', signal_est2, delimiter=',')
np.savetxt('signal_est_post2.csv', signal_est_post2, delimiter=',')



# ### Well seperated 0.5N(2, 3)+0.5N(-2, 3) with Pure background
#well_seperated+pure
np.random.seed(2333)
random.seed(2333)
large_data3 = np.zeros((128, 128, 100))
large_signals3 = np.zeros((128, 128, 100))
for i in range(100):
    large_result3 = data_gen(signal_prior, signal_weights1, signal_dist_well)
    large_data3[0:128, 0:128, i] = large_result3['z']
    large_signals3[0:128, 0:128, i] = large_result3['flips']

for i in range(0, 100):
    print(i)
    lfdr = local_fdr(large_data3[0:128, 0:128, i].flatten(), deg=10, nbins=30)

fnp3 = np.zeros((100,7))
fdr3 = np.zeros((100,7))
fm3 = np.zeros((100,7))
signal_est3 = np.zeros(100)
signal_est_post3 = np.zeros(100)
for i in range(0, 100):
    results_3 = simulation_run(large_data3[0:128, 0:128, i], large_signals3[0:128, 0:128, i], 0.05, 0.1)
    fnp3[i,:] = results_3['fnp']
    fdr3[i,:] = results_3['fdr']
    fm3[i,:] = results_3['fm']
    signal_est3[i] = results_3['signal_est']
    signal_est_post3[i] = results_3['signal_est_post']


np.savetxt('fnp3.csv', fnp3, delimiter=',')
np.savetxt('fdr3.csv', fdr3, delimiter=',')
np.savetxt('fm3.csv', fm3, delimiter=',')
np.savetxt('signal_est3.csv', signal_est3, delimiter=',')
np.savetxt('signal_est_post3.csv', signal_est_post3, delimiter=',')


# ### Well seperated 0.5N(2, 3)+0.5N(-2, 3) with Mixed background
#well-seperated+mixed
np.random.seed(2333)
random.seed(2333)
large_data4 = np.zeros((128, 128, 100))
large_signals4 = np.zeros((128, 128, 100))
for i in range(100):
    large_result4 = data_gen(signal_prior, signal_weights3, signal_dist_well)
    large_data4[0:128, 0:128, i] = large_result4['z']
    large_signals4[0:128, 0:128, i] = large_result4['flips']

for i in range(0, 100):
    print(i)
    lfdr = local_fdr(large_data4[0:128, 0:128, i].flatten())   

fnp4 = np.zeros((100,7))
fdr4 = np.zeros((100,7))
fm4 = np.zeros((100,7))
signal_est4 = np.zeros(100)
signal_est_post4 = np.zeros(100)
for i in range(0, 100):
    results_4 = simulation_run(large_data4[0:128, 0:128, i], large_signals4[0:128, 0:128, i], 0.05, 0.1)
    fnp4[i,:] = results_4['fnp']
    fdr4[i,:] = results_4['fdr']
    fm4[i,:] = results_4['fm']
    signal_est4[i] = results_4['signal_est']
    signal_est_post4[i] = results_4['signal_est_post']


np.savetxt('fnp4.csv', fnp4, delimiter=',')
np.savetxt('fdr4.csv', fdr4, delimiter=',')
np.savetxt('fm4.csv', fm4, delimiter=',')
np.savetxt('signal_est4.csv', signal_est4, delimiter=',')
np.savetxt('signal_est_post4.csv', signal_est_post4, delimiter=',')
