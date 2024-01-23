# Import package
import pygfl, sys, os, time, ramdom, matplotlib, math
import matplotlib.pylab as plt
import numpy as np
import nibabel as nib
import scipy.stats as st
from random import sample
from collections import defaultdict
from statsmodels.stats.multitest import local_fdr
from smoothfdr import smooth_fdr, local_agg_fdr

# BH-FDR function
## method function
def benjamini_hochberg(z, fdr, mu0=0., sigma0=1.):
    z_shape = z.shape if len(z.shape) > 1 else None
    if z_shape is not None:
        z = z.flatten()
    p = p_value(z, mu0=mu0, sigma0=sigma0)
    p_orders = np.argsort(p)
    discoveries = []
    m = float(len(p_orders))
    for k, s in enumerate(p_orders):
        if p[s] <= (k+1) / m * fdr:
            discoveries.append(s)
        else:
            break
    discoveries = np.array(discoveries)
    if (z_shape is not None) and len(discoveries)>0:
        x = np.zeros(z.shape)
        x[discoveries] = 1
        discoveries = np.where(x.reshape(z_shape) == 1)
    return discoveries

## wrapper function for simulation
def bh_run(data, fdr_level):
    dis_bh = benjamini_hochberg(data, fdr_level, mu0=0., sigma0=1.)
    results_bh = np.zeros(data.shape)
    if np.any(dis_bh):
        results_bh[dis_bh] = 1
    return(results_bh)


# FDR Smoothing
## Import the Edge file
def load_edges(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        edges = defaultdict(list)
        for line in reader:
            nodes = [int(x) for x in line]
            for n1,n2 in zip(nodes[:-1], nodes[1:]):
                edges[n1].append(n2)
                edges[n2].append(n1)
    return edges

## replace the path by that of edge file
edges = load_edges('/Users/yifeihu/Desktop/simulation_test/edges.csv')

## wrapper function for simulation
def fdrs_run(z, fdr_level):
    results_fdrs = smooth_fdr(z, fdr_level, verbose=5, missing_val=0)
    return {'de': results_fdrs['discoveries'],
			'lfdr': 1-results_fdrs['posteriors'],
			'post': results_fdrs['posteriors'],
			'c': results_fdrs['priors'],
			'1-c': 1-results_fdrs['priors']}

# FDRL
## wrapper function for simulation
def fdrl_run(data, fdr_level):
    p_values = 2*(1.0 - st.norm.cdf(np.abs(data)))
    p_values_flat = p_values.flatten()
    dis_fdrl = local_agg_fdr(p_values_flat, edges, fdr_level, lmbda = 0.2)
    results_fdrl = np.zeros(p_values_flat.shape)
    results_fdrl[dis_fdrl] = 1
    results_fdrl = results_fdrl.reshape(data.shape)
    return(results_fdrl)


# MDR

## MDR procedure
def MDR_proc(lfdr, eps, alpha):
    n = lfdr.size
    de = np.zeros(n)
    nmp = n*eps*alpha
    lfdr_f = lfdr.flatten()
    tor = 1-lfdr_f
    tor.sort()
    cumsum_tor = np.cumsum(tor)
    k = np.sum(cumsum_tor <= nmp)
    threshold = 1-tor[k-1]
    reject = (lfdr_f<=threshold)
    accept = (lfdr_f>threshold)
    de[reject] = 1
    reject = reject.reshape(lfdr.shape)
    accept = accept.reshape(lfdr.shape)
    de = de.reshape(lfdr.shape)
    return {'nr': k,
			'th': threshold,
			're': reject,
			'ac': accept,
			'de': de}

# JC estimator
def epsest_func(x, u, sigma):
    z = (x-u)/sigma
    xi = np.arange(101)/100
    tmax = np.sqrt(math.log(x.size))
    tt = np.arange(0, (tmax+0.1), 0.1)
    epsest = np.zeros(tt.size)
    for j in range(tt.size):
        t = tt[j]
        f = t*xi
        f = np.exp(f*f/2)
        w = (1-np.abs(xi))
        co = 0*xi
        for i in range(101):
            co[i] = np.mean(np.cos(t*xi[i]*z))
        epshat = 1 - np.sum(w*f*co)/np.sum(w)
        epsest[j] = epshat
    return np.max(epsest)

# wrapper function for simulation
def MDR_run(z, epsilon):
    z_f = z.flatten()
    lfdr = local_fdr(z_f, deg=10, nbins=30)
    lfdr = lfdr.reshape(z.shape)
    results = MDR_proc(lfdr, epsest_func(z_f, 0, 1), epsilon)
    return results['de']

# AFNC
## wrapper function for simulation
def AFNC_run(z, epsilon):
    np.random.seed(2333)
    n = z.size
    z_f = z.flatten()
    s = int(epsest_func(z_f, 0, 1) * n)
    pval = p_value(z_f)
    p_order = p_value(z_f)
    p_order.sort()
    p_order_s = p_order[s:n]
    numNul = n-s
    quant = np.zeros(numNul)
    for i in range(numNul):
        quant[i] = st.beta.ppf(epsilon, (i+1), (numNul-i))
    if s==0:
        jstar = 0
    else:
        t1 = np.sum(pval<epsilon/numNul)
        if s <= t1:
            jhat = 0
        else:
            if (p_order_s <= quant).sum() > 0:
                jhat = min(np.min(np.where(p_order_s <= quant))+1, numNul)
            else:
                jhat = 0
        jstar = min(s+jhat, 16383)
    t = p_order[jstar]
    reject = (pval<=t)
    de = np.zeros(n)
    de[reject] = 1
    de = de.reshape(z.shape)
    return(de)

# SMDR
## using JC estimator
def smdr_run1(z, lfdr, epsilon):
    results = MDR_proc(lfdr, epsest_func(z, 0, 1), epsilon)
    return results['de']

## with new estimator
def smdr_run2(est, lfdr, epsilon):
    results = MDR_proc(lfdr, est, epsilon)
    return results['de']

# Data generating function
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

# Metrics calculation functions
def fnp_cal(de, truth):
    fnp = np.logical_and(truth == 1, de == 0).sum() / float((truth == 1).sum())
    return fnp

def fdr_cal(de, truth):
    fdr = np.logical_and(truth == 0, de == 1).sum() / max(1, float((de == 1).sum()))
    return fdr

def fm_cal(de, truth):
    tp = np.logical_and(truth == 1, de == 1).sum()
    fp = np.logical_and(truth == 0, de == 1).sum()
    fn = np.logical_and(truth == 1, de == 0).sum()
    fm = math.sqrt((tp/(tp+fp))*(tp/(tp+fn)))
    return fm

# Simulation functions
def simulation_run_new(z, signals, fdr_level, epsilon):
    # BH-FDR(0.05)
    de_bh1 = bh_run(z, 0.05)
    # BH-FDR(0.1)
    de_bh2 = bh_run(z, 0.1)
    # FDRL(0.05)
    de_fdrl1 = fdrl_run(z, 0.05)
    # FDRL(0.1)
    de_fdrl2 = fdrl_run(z, 0.1)
    # FDRS(0.05)
    results_fdrs1 = fdrs_run(z, 0.05)
    de_fdrs1 = results_fdrs1['de']
    lfdr_fdrs = results_fdrs1['lfdr']
    # FDRS(0.1)
    results_fdrs2 = fdrs_run(z, 0.1)
    de_fdrs2 = results_fdrs2['de']
    # MDR
    de_mdr = MDR_run(z, 0.1)
    # AFNC
    de_afnc = AFNC_run(z, 0.1)
    # SAFN (with JC estimator)
    de_new = smdr_run1(z, lfdr_fdrs, epsilon)
    # SMDR(0.05)
    de_new2 = smdr_run2(results_fdrs1['post'].sum()/(128*128), lfdr_fdrs, 0.05)
    # SMDR(0.1)
    de_new3 = smdr_run2(results_fdrs1['post'].sum()/(128*128), lfdr_fdrs, 0.1)

    
    signal_est = round(epsest_func(z, 0, 1)*z.size)
    signal_est_prior = round(results_fdrs1['c'].sum())
    signal_est_post = round(results_fdrs1['post'].sum())
    
    fnp = np.zeros(11)
    fdr = np.zeros(11)
    fm = np.zeros(11)
    f1 = np.zeros(11)
    
    fnp[0] = fnp_cal(de_bh1, signals)
    fnp[1] = fnp_cal(de_bh2, signals)
    fnp[2] = fnp_cal(de_fdrl1, signals)
    fnp[3] = fnp_cal(de_fdrl2, signals)
    fnp[4] = fnp_cal(de_fdrs1, signals)
    fnp[5] = fnp_cal(de_fdrs2, signals)
    fnp[6] = fnp_cal(de_mdr, signals)
    fnp[7] = fnp_cal(de_afnc, signals)
    fnp[8] = fnp_cal(de_new, signals)
    fnp[9] = fnp_cal(de_new2, signals)
    fnp[10] = fnp_cal(de_new3, signals)

    fdr[0] = fdr_cal(de_bh1, signals)
    fdr[1] = fdr_cal(de_bh2, signals)
    fdr[2] = fdr_cal(de_fdrl1, signals)
    fdr[3] = fdr_cal(de_fdrl2, signals)
    fdr[4] = fdr_cal(de_fdrs1, signals)
    fdr[5] = fdr_cal(de_fdrs2, signals)
    fdr[6] = fdr_cal(de_mdr, signals)
    fdr[7] = fdr_cal(de_afnc, signals)
    fdr[8] = fdr_cal(de_new, signals)
    fdr[9] = fdr_cal(de_new2, signals)
    fdr[10] = fdr_cal(de_new3, signals)

    fm[0] = fm_cal(de_bh1, signals)
    fm[1] = fm_cal(de_bh2, signals)
    fm[2] = fm_cal(de_fdrl1, signals)
    fm[3] = fm_cal(de_fdrl2, signals)
    fm[4] = fm_cal(de_fdrs1, signals)
    fm[5] = fm_cal(de_fdrs2, signals)
    fm[6] = fm_cal(de_mdr, signals)
    fm[7] = fm_cal(de_afnc, signals)
    fm[8] = fm_cal(de_new, signals)
    fm[9] = fm_cal(de_new2, signals)
    fm[10] = fm_cal(de_new3, signals)

    f1[0] = f1_cal(de_bh1, signals)
    f1[1] = f1_cal(de_bh2, signals)
    f1[2] = f1_cal(de_fdrl1, signals)
    f1[3] = f1_cal(de_fdrl2, signals)
    f1[4] = f1_cal(de_fdrs1, signals)
    f1[5] = f1_cal(de_fdrs2, signals)
    f1[6] = f1_cal(de_mdr, signals)
    f1[7] = f1_cal(de_afnc, signals)
    f1[8] = f1_cal(de_new, signals)
    f1[9] = f1_cal(de_new2, signals)
    f1[10] = f1_cal(de_new3, signals)

    return {'fnp': fnp,
			'fdr': fdr,
			'fm': fm,
            'f1': f1,
            'de_bh1': de_bh1,
            'de_bh2': de_bh2,
            'de_fdrl1': de_fdrl1,
            'de_fdrl2': de_fdrl2,
            'de_fdrs1': de_fdrs1,
            'de_fdrs2': de_fdrs2,
            'de_mdr': de_mdr,
            'de_afnc': de_afnc,
            'de_new': de_new,
            'de_new2': de_new2,
            'de_new3': de_new3,
            'signal_est': signal_est,
            'signal_est_prior': signal_est_prior,
            'signal_est_post': signal_est_post}

# Simulation
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

# generate signal weight maps
## saturated+pure
signal_weights1 = signal_prior.copy()

# generate two signal distributions
## Heteroscadastic
np.random.seed(2333)
signal_dist_poorly3 = list(np.random.normal(loc=0, scale=3, size=128*128))
signal_dist_poorly2 = list(np.random.normal(loc=0, scale=2, size=128*128))

## Heterogeneous
np.random.seed(2333)
signal_dist_well = np.zeros((128*128))
for i in range(128*128):
    flip_well = np.random.random(size=1) > 0.5
    if flip_well: signal_dist_well[i] = np.random.normal(loc=-2.5, scale=1, size=1)
    else: signal_dist_well[i] = np.random.normal(loc=2.5, scale=1, size=1)
signal_dist_well = list(signal_dist_well)


signal_dist_well2 = np.zeros((128*128))
for i in range(128*128):
    flip_well2 = np.random.random(size=1) > 0.5
    if flip_well2: signal_dist_well2[i] = np.random.normal(loc=-1, scale=1, size=1)
    else: signal_dist_well2[i] = np.random.normal(loc=1, scale=1, size=1)
signal_dist_well2 = list(signal_dist_well2)


signal_dist_well3 = np.zeros((128*128))
for i in range(128*128):
    flip_well3 = np.random.random(size=1) > 0.5
    if flip_well3: signal_dist_well3[i] = np.random.normal(loc=-3, scale=1, size=1)
    else: signal_dist_well3[i] = np.random.normal(loc=3, scale=1, size=1)
signal_dist_well3 = list(signal_dist_well3)


signal_dist_well4 = np.zeros((128*128))
for i in range(128*128):
    flip_well4 = np.random.random(size=1) > 0.5
    if flip_well4: signal_dist_well4[i] = np.random.normal(loc=-2, scale=1, size=1)
    else: signal_dist_well4[i] = np.random.normal(loc=2, scale=1, size=1)
signal_dist_well4 = list(signal_dist_well4)

########################

## One-time example
### Setting 1
np.random.seed(2333)
random.seed(2333)
data = data_gen(signal_prior, signal_weights1, signal_dist_poorly3)
z = data['z']
signals = data['flips']
results = simulation_run_new(z, signals, 0.05, 0.1)
plt.rcParams["figure.figsize"] = (20,15)
fig, ax = plt.subplots(3,4)
ax[0,0].imshow(signals, cmap='gray_r', vmin=0, vmax=1)
ax[0,0].set_title('Signal region', fontsize=20)
ax[0,1].imshow(z, cmap='gray_r', vmin=0, vmax=1)
ax[0,1].set_title('Z statistics', fontsize=20)
ax[0,2].imshow(results['de_bh1'], cmap='gray_r', vmin=0, vmax=1)
ax[0,2].set_title('BH-FDR(alpha=0.05)', fontsize=20)
ax[0,3].imshow(results['de_bh2'], cmap='gray_r', vmin=0, vmax=1)
ax[0,3].set_title('BH-FDR(alpha=0.1)', fontsize=20)
ax[1,0].imshow(results['de_fdrl1'], cmap='gray_r', vmin=0, vmax=1)
ax[1,0].set_title('FDRL(alpha=0.05)', fontsize=20)
ax[1,1].imshow(results['de_fdrl2'], cmap='gray_r', vmin=0, vmax=1)
ax[1,1].set_title('FDRL(alpha=0.1)', fontsize=20)
ax[1,2].imshow(results['de_fdrs1'], cmap='gray_r', vmin=0, vmax=1)
ax[1,2].set_title('FDRS(alpha=0.05)', fontsize=20)
ax[1,3].imshow(results['de_fdrs2'], cmap='gray_r', vmin=0, vmax=1)
ax[1,3].set_title('FDRS(alpha=0.1)', fontsize=20)
ax[2,0].imshow(results['de_afnc'], cmap='gray_r', vmin=0, vmax=1)
ax[2,0].set_title('AFNC(beta=0.1)', fontsize=20)
ax[2,1].imshow(results['de_mdr'], cmap='gray_r', vmin=0, vmax=1)
ax[2,1].set_title('CS-MDR(beta=0.1)', fontsize=20)
ax[2,2].imshow(results['de_new3'], cmap='gray_r', vmin=0, vmax=1)
ax[2,2].set_title('SMDR(beta=0.1)', fontsize=20)
ax[2,3].imshow(results['de_new2'], cmap='gray_r', vmin=0, vmax=1)
ax[2,3].set_title('SMDR(beta=0.05)', fontsize=20)

### Setting 2
np.random.seed(2333)
random.seed(2333)
data2 = data_gen(signal_prior, signal_weights1, signal_dist_poorly2)
z2 = data2['z']
signals2 = data2['flips']
results2 = simulation_run_new(z2, signals2, 0.05, 0.1)
plt.rcParams["figure.figsize"] = (20,15)
fig, ax = plt.subplots(3,4)
ax[0,0].imshow(signals2, cmap='gray_r', vmin=0, vmax=1)
ax[0,0].set_title('Signal region', fontsize=20)
ax[0,1].imshow(z2, cmap='gray_r', vmin=0, vmax=1)
ax[0,1].set_title('Z statistics', fontsize=20)
ax[0,2].imshow(results2['de_bh1'], cmap='gray_r', vmin=0, vmax=1)
ax[0,2].set_title('BH-FDR(alpha=0.05)', fontsize=20)
ax[0,3].imshow(results2['de_bh2'], cmap='gray_r', vmin=0, vmax=1)
ax[0,3].set_title('BH-FDR(alpha=0.1)', fontsize=20)
ax[1,0].imshow(results2['de_fdrl1'], cmap='gray_r', vmin=0, vmax=1)
ax[1,0].set_title('FDRL(alpha=0.05)', fontsize=20)
ax[1,1].imshow(results2['de_fdrl2'], cmap='gray_r', vmin=0, vmax=1)
ax[1,1].set_title('FDRL(alpha=0.1)', fontsize=20)
ax[1,2].imshow(results2['de_fdrs1'], cmap='gray_r', vmin=0, vmax=1)
ax[1,2].set_title('FDRS(alpha=0.05)', fontsize=20)
ax[1,3].imshow(results2['de_fdrs2'], cmap='gray_r', vmin=0, vmax=1)
ax[1,3].set_title('FDRS(alpha=0.1)', fontsize=20)
ax[2,0].imshow(results2['de_afnc'], cmap='gray_r', vmin=0, vmax=1)
ax[2,0].set_title('AFNC(beta=0.1)', fontsize=20)
ax[2,1].imshow(results2['de_mdr'], cmap='gray_r', vmin=0, vmax=1)
ax[2,1].set_title('CS-MDR(beta=0.1)', fontsize=20)
ax[2,2].imshow(results2['de_new3'], cmap='gray_r', vmin=0, vmax=1)
ax[2,2].set_title('SMDR(beta=0.1)', fontsize=20)
ax[2,3].imshow(results2['de_new2'], cmap='gray_r', vmin=0, vmax=1)
ax[2,3].set_title('SMDR(beta=0.05)', fontsize=20)

### Setting 3
np.random.seed(2333)
random.seed(2333)
data3 = data_gen(signal_prior, signal_weights1, signal_dist_well)
z3 = data3['z']
signals3 = data3['flips']
results3 = simulation_run_new(z3, signals3, 0.05, 0.1)
plt.rcParams["figure.figsize"] = (20,15)
fig, ax = plt.subplots(3,4)
ax[0,0].imshow(signals3, cmap='gray_r', vmin=0, vmax=1)
ax[0,0].set_title('Signal region')
ax[0,1].imshow(z3, cmap='gray_r', vmin=0, vmax=1)
ax[0,1].set_title('Raw data')
ax[0,2].imshow(results3['de_bh1'], cmap='gray_r', vmin=0, vmax=1)
ax[0,2].set_title('BH-FDR(alpha=0.05)')
ax[0,3].imshow(results3['de_bh2'], cmap='gray_r', vmin=0, vmax=1)
ax[0,3].set_title('BH-FDR(alpha=0.1)')
ax[1,0].imshow(results3['de_fdrl1'], cmap='gray_r', vmin=0, vmax=1)
ax[1,0].set_title('FDRL(alpha=0.05)')
ax[1,1].imshow(results3['de_fdrl2'], cmap='gray_r', vmin=0, vmax=1)
ax[1,1].set_title('FDRL(alpha=0.1)')
ax[1,2].imshow(results3['de_fdrs1'], cmap='gray_r', vmin=0, vmax=1)
ax[1,2].set_title('FDRS(alpha=0.05)')
ax[1,3].imshow(results3['de_fdrs2'], cmap='gray_r', vmin=0, vmax=1)
ax[1,3].set_title('FDRS(alpha=0.1)')
ax[2,0].imshow(results3['de_afnc'], cmap='gray_r', vmin=0, vmax=1)
ax[2,0].set_title('AFNC(beta=0.1)')
ax[2,1].imshow(results3['de_mdr'], cmap='gray_r', vmin=0, vmax=1)
ax[2,1].set_title('MDR(beta=0.1)')
ax[2,2].imshow(results3['de_new3'], cmap='gray_r', vmin=0, vmax=1)
ax[2,2].set_title('SMDR(beta=0.1)')
ax[2,3].imshow(results3['de_new2'], cmap='gray_r', vmin=0, vmax=1)
ax[2,3].set_title('SMDR(beta=0.05)')

### Setting 4
np.random.seed(2333)
random.seed(2333)
data4 = data_gen(signal_prior, signal_weights1, signal_dist_well2)
z4 = data4['z']
signals4 = data4['flips']
results4 = simulation_run_new(z4, signals4, 0.05, 0.1)
plt.rcParams["figure.figsize"] = (20,15)
fig, ax = plt.subplots(3,4)
ax[0,0].imshow(signals4, cmap='gray_r', vmin=0, vmax=1)
ax[0,0].set_title('Signal region', fontsize=20)
ax[0,1].imshow(z4, cmap='gray_r', vmin=0, vmax=1)
ax[0,1].set_title('Z statistics', fontsize=20)
ax[0,2].imshow(results4['de_bh1'], cmap='gray_r', vmin=0, vmax=1)
ax[0,2].set_title('BH-FDR(alpha=0.05)', fontsize=20)
ax[0,3].imshow(results4['de_bh2'], cmap='gray_r', vmin=0, vmax=1)
ax[0,3].set_title('BH-FDR(alpha=0.1)', fontsize=20)
ax[1,0].imshow(results4['de_fdrl1'], cmap='gray_r', vmin=0, vmax=1)
ax[1,0].set_title('FDRL(alpha=0.05)', fontsize=20)
ax[1,1].imshow(results4['de_fdrl2'], cmap='gray_r', vmin=0, vmax=1)
ax[1,1].set_title('FDRL(alpha=0.1)', fontsize=20)
ax[1,2].imshow(results4['de_fdrs1'], cmap='gray_r', vmin=0, vmax=1)
ax[1,2].set_title('FDRS(alpha=0.05)', fontsize=20)
ax[1,3].imshow(results4['de_fdrs2'], cmap='gray_r', vmin=0, vmax=1)
ax[1,3].set_title('FDRS(alpha=0.1)', fontsize=20)
ax[2,0].imshow(results4['de_afnc'], cmap='gray_r', vmin=0, vmax=1)
ax[2,0].set_title('AFNC(beta=0.1)', fontsize=20)
ax[2,1].imshow(results4['de_mdr'], cmap='gray_r', vmin=0, vmax=1)
ax[2,1].set_title('CS-MDR(beta=0.1)', fontsize=20)
ax[2,2].imshow(results4['de_new3'], cmap='gray_r', vmin=0, vmax=1)
ax[2,2].set_title('SMDR(beta=0.1)', fontsize=20)
ax[2,3].imshow(results4['de_new2'], cmap='gray_r', vmin=0, vmax=1)
ax[2,3].set_title('SMDR(beta=0.05)', fontsize=20)

### Setting 5
np.random.seed(2333)
random.seed(2333)
data5 = data_gen2(4, 2)
z5 = data5['z']
signals5 = data5['flips']
results5 = simulation_run_new(z5, signals5, 0.05, 0.1)
plt.rcParams["figure.figsize"] = (20,15)
fig, ax = plt.subplots(3,4)
ax[0,0].imshow(signals5, cmap='gray_r', vmin=0, vmax=1)
ax[0,0].set_title('Signal region', fontsize=20)
ax[0,1].imshow(z5, cmap='gray_r', vmin=0, vmax=1)
ax[0,1].set_title('Z statistics', fontsize=20)
ax[0,2].imshow(results5['de_bh1'], cmap='gray_r', vmin=0, vmax=1)
ax[0,2].set_title('BH-FDR(alpha=0.05)', fontsize=20)
ax[0,3].imshow(results5['de_bh2'], cmap='gray_r', vmin=0, vmax=1)
ax[0,3].set_title('BH-FDR(alpha=0.1)', fontsize=20)
ax[1,0].imshow(results5['de_fdrl1'], cmap='gray_r', vmin=0, vmax=1)
ax[1,0].set_title('FDRL(alpha=0.05)', fontsize=20)
ax[1,1].imshow(results5['de_fdrl2'], cmap='gray_r', vmin=0, vmax=1)
ax[1,1].set_title('FDRL(alpha=0.1)', fontsize=20)
ax[1,2].imshow(results5['de_fdrs1'], cmap='gray_r', vmin=0, vmax=1)
ax[1,2].set_title('FDRS(alpha=0.05)', fontsize=20)
ax[1,3].imshow(results5['de_fdrs2'], cmap='gray_r', vmin=0, vmax=1)
ax[1,3].set_title('FDRS(alpha=0.1)', fontsize=20)
ax[2,0].imshow(results5['de_afnc'], cmap='gray_r', vmin=0, vmax=1)
ax[2,0].set_title('AFNC(beta=0.1)', fontsize=20)
ax[2,1].imshow(results5['de_mdr'], cmap='gray_r', vmin=0, vmax=1)
ax[2,1].set_title('MDR(beta=0.1)', fontsize=20)
ax[2,2].imshow(results5['de_new3'], cmap='gray_r', vmin=0, vmax=1)
ax[2,2].set_title('SMDR(beta=0.1)', fontsize=20)
ax[2,3].imshow(results5['de_new2'], cmap='gray_r', vmin=0, vmax=1)
ax[2,3].set_title('SMDR(beta=0.05)', fontsize=20)

### Setting 6
np.random.seed(2333)
random.seed(2333)
data6 = data_gen2(2, 1)
z6 = data6['z']
signals6 = data6['flips']
results6 = simulation_run_new(z6, signals6, 0.05, 0.1)
plt.rcParams["figure.figsize"] = (20,15)
fig, ax = plt.subplots(3,4)
ax[0,0].imshow(signals6, cmap='gray_r', vmin=0, vmax=1)
ax[0,0].set_title('Signal region', fontsize=20)
ax[0,1].imshow(z6, cmap='gray_r', vmin=0, vmax=1)
ax[0,1].set_title('Z statistics', fontsize=20)
ax[0,2].imshow(results6['de_bh1'], cmap='gray_r', vmin=0, vmax=1)
ax[0,2].set_title('BH-FDR(alpha=0.05)', fontsize=20)
ax[0,3].imshow(results6['de_bh2'], cmap='gray_r', vmin=0, vmax=1)
ax[0,3].set_title('BH-FDR(alpha=0.1)', fontsize=20)
ax[1,0].imshow(results6['de_fdrl1'], cmap='gray_r', vmin=0, vmax=1)
ax[1,0].set_title('FDRL(alpha=0.05)', fontsize=20)
ax[1,1].imshow(results6['de_fdrl2'], cmap='gray_r', vmin=0, vmax=1)
ax[1,1].set_title('FDRL(alpha=0.1)', fontsize=20)
ax[1,2].imshow(results6['de_fdrs1'], cmap='gray_r', vmin=0, vmax=1)
ax[1,2].set_title('FDRS(alpha=0.05)', fontsize=20)
ax[1,3].imshow(results6['de_fdrs2'], cmap='gray_r', vmin=0, vmax=1)
ax[1,3].set_title('FDRS(alpha=0.1)', fontsize=20)
ax[2,0].imshow(results6['de_afnc'], cmap='gray_r', vmin=0, vmax=1)
ax[2,0].set_title('AFNC(beta=0.1)', fontsize=20)
ax[2,1].imshow(results6['de_mdr'], cmap='gray_r', vmin=0, vmax=1)
ax[2,1].set_title('MDR(beta=0.1)', fontsize=20)
ax[2,2].imshow(results6['de_new3'], cmap='gray_r', vmin=0, vmax=1)
ax[2,2].set_title('SMDR(beta=0.1)', fontsize=20)
ax[2,3].imshow(results6['de_new2'], cmap='gray_r', vmin=0, vmax=1)
ax[2,3].set_title('SMDR(beta=0.05)', fontsize=20)

## Large sample simulation
########################
## Simulation Function ##
########################
def data_gen1(prior, signal_weights, signal_dist):
    # Generate spatial correlated noise
    data_e = np.zeros((128, 128))
    data_epsilon = np.zeros((128, 128))
    for i in range(128):
        for j in range(128):
            data_e[i, j] = np.random.normal(loc=0, scale=1, size=1)    
    for i in range(128):
        for j in range(128):
            k = 1
            data_epsilon[i, j] += data_e[i, j]
            if i>0:
                data_epsilon[i, j] += data_e[i-1, j]
                k += 1
            if j>0:
                data_epsilon[i, j] += data_e[i, j-1]
                k += 1
            if i<127:
                data_epsilon[i, j] += data_e[i+1, j]
                k += 1
            if j<127:
                data_epsilon[i, j] += data_e[i, j+1]
                k += 1
            data_epsilon[i, j] /= math.sqrt(k)
    
    # Generate signal area
    samples = np.zeros((128, 128))
    flips = np.random.random(size=signal_weights.shape) < signal_weights
    for i in range(128):
        for j in range(128):
            if flips[i, j]: 
                samples[i, j] = sample(signal_dist, 1) + data_epsilon[i, j]
            else:
                samples[i, j] = data_epsilon[i, j]
    signals = np.zeros((128, 128))
    signals[np.logical_and(prior == 1, flips==1)] = 1
    return {'z': samples,
			'signals': signals,
            'flips': flips}


def data_gen2(mu1, mu2):
    data_mu = np.zeros((128, 128))
    data_signal = np.zeros((128, 128))
    data_e = np.zeros((128, 128))
    for i in range(128):
        for j in range(128):
            data_e[i, j] = np.random.normal(loc=0, scale=1, size=1)
            if i>20 and i<=50 and j>20 and j<=50:
                data_mu[i, j] = mu1
                data_signal[i, j] = 1
            if i>85 and i<=100 and j>85 and j<=100:
                data_mu[i, j] = mu2
                data_signal[i, j] = 1
                
    data_epsilon = np.zeros((128, 128))
    for i in range(128):
        for j in range(128):
            k = 1
            data_epsilon[i, j] += data_e[i, j]
            if i>0:
                data_epsilon[i, j] += data_e[i-1, j]
                k += 1
            if j>0:
                data_epsilon[i, j] += data_e[i, j-1]
                k += 1
            if i<127:
                data_epsilon[i, j] += data_e[i+1, j]
                k += 1
            if j<127:
                data_epsilon[i, j] += data_e[i, j+1]
                k += 1
            data_epsilon[i, j] /= math.sqrt(k)
    samples = data_mu + data_epsilon
    return {'z': samples,
			'signals': data_signal,
            'flips': data_signal}


### Heteroscadastic 2
np.random.seed(2333)
random.seed(2333)
large_data0 = np.zeros((128, 128, 100))
large_signals0 = np.zeros((128, 128, 100))
for i in range(100):
    large_result0 = data_gen1(signal_prior, signal_weights1, signal_dist_poorly2)
    large_data0[0:128, 0:128, i] = large_result0['z']
    large_signals0[0:128, 0:128, i] = large_result0['flips']
fnp0 = np.zeros((100,11))
fdr0 = np.zeros((100,11))
fm0 = np.zeros((100,11))
signal_est0 = np.zeros(100)
signal_est_prior0 = np.zeros(100)
signal_est_post0 = np.zeros(100)
for i in range(0, 100):
    results_0 = simulation_run_new(large_data0[0:128, 0:128, i], large_signals0[0:128, 0:128, i], 0.05, 0.1)
    fnp0[i,:] = results_0['fnp']
    fdr0[i,:] = results_0['fdr']
    fm0[i,:] = results_0['fm']
    signal_est0[i] = results_0['signal_est']
    signal_est_prior0[i] = results_0['signal_est_prior']
    signal_est_post0[i] = results_0['signal_est_post']

np.mean(fnp0, axis=0), np.std(fnp0, axis=0)
np.mean(fdr0, axis=0), np.std(fdr0, axis=0)
np.mean(fm0, axis=0), np.std(fm0, axis=0)

pd.DataFrame(fnp0).to_csv('heteroscadastic_sc_2_fnp.csv')
pd.DataFrame(fdr0).to_csv('heteroscadastic_sc_2_fdr.csv')
pd.DataFrame(fm0).to_csv('heteroscadastic_sc_2_fm.csv')

signal_est_jc_0 = np.zeros(100)
for i in range(0, 100):
    z_f = large_data0[0:128, 0:128, i].flatten()
    signal_est_jc_0[i] = epsest_func(z_f, 0, 1)
    
signal_est_jc_0 = signal_est_jc_0*128*128
ratio_jc_0 = signal_est_jc_0 / 1686
np.mean(ratio_jc_0, axis=0), np.std(ratio_jc_0, axis=0)
ratio0 = signal_est_post0 / 1686
np.mean(ratio0, axis=0), np.std(ratio0, axis=0)


### Heteroscadastic 3
np.random.seed(2333)
random.seed(2333)
large_data1 = np.zeros((128, 128, 100))
large_signals1 = np.zeros((128, 128, 100))
for i in range(100):
    large_result1 = data_gen1(signal_prior, signal_weights1, signal_dist_poorly3)
    large_data1[0:128, 0:128, i] = large_result1['z']
    large_signals1[0:128, 0:128, i] = large_result1['flips']
fnp1 = np.zeros((100,11))
fdr1 = np.zeros((100,11))
fm1 = np.zeros((100,11))
signal_est1 = np.zeros(100)
signal_est_prior1 = np.zeros(100)
signal_est_post1 = np.zeros(100)
for i in range(0, 100):
    results_1 = simulation_run_new(large_data1[0:128, 0:128, i], large_signals1[0:128, 0:128, i], 0.05, 0.1)
    fnp1[i,:] = results_1['fnp']
    fdr1[i,:] = results_1['fdr']
    fm1[i,:] = results_1['fm']
    signal_est1[i] = results_1['signal_est']
    signal_est_prior1[i] = results_1['signal_est_prior']
    signal_est_post1[i] = results_1['signal_est_post']

np.mean(fnp1, axis=0), np.std(fnp1, axis=0)
np.mean(fdr1, axis=0), np.std(fdr1, axis=0)
np.mean(fm1, axis=0), np.std(fm1, axis=0)

pd.DataFrame(fnp1).to_csv('heteroscadastic_sc_3_fnp.csv')
pd.DataFrame(fdr1).to_csv('heteroscadastic_sc_3_fdr.csv')
pd.DataFrame(fm1).to_csv('heteroscadastic_sc_3_fm.csv')

signal_est_jc_1 = np.zeros(100)
for i in range(0, 100):
    z_f = large_data1[0:128, 0:128, i].flatten()
    signal_est_jc_1[i] = epsest_func(z_f, 0, 1)
    
signal_est_jc_1 = signal_est_jc_1*128*128
ratio_jc_1 = signal_est_jc_1 / 1686
np.mean(ratio_jc_1, axis=0), np.std(ratio_jc_1, axis=0)
ratio1 = signal_est_post1 / 1686
np.mean(ratio1, axis=0), np.std(ratio1, axis=0)

### Heterogeneous 2.5
np.random.seed(2333)
random.seed(2333)
large_data2 = np.zeros((128, 128, 100))
large_signals2 = np.zeros((128, 128, 100))
for i in range(100):
    large_result2 = data_gen1(signal_prior, signal_weights1, signal_dist_well)
    large_data2[0:128, 0:128, i] = large_result2['z']
    large_signals2[0:128, 0:128, i] = large_result2['flips']
fnp2 = np.zeros((100,11))
fdr2 = np.zeros((100,11))
fm2 = np.zeros((100,11))
signal_est2 = np.zeros(100)
signal_est_prior2 = np.zeros(100)
signal_est_post2 = np.zeros(100)
for i in range(0, 100):
    results_2 = simulation_run_new(large_data2[0:128, 0:128, i], large_signals2[0:128, 0:128, i], 0.05, 0.1)
    fnp2[i,:] = results_2['fnp']
    fdr2[i,:] = results_2['fdr']
    fm2[i,:] = results_2['fm']
    signal_est2[i] = results_2['signal_est']
    signal_est_prior2[i] = results_2['signal_est_prior']
    signal_est_post2[i] = results_2['signal_est_post']
pd.DataFrame(fnp2).to_csv('heterogeneous_sc_fnp.csv')
pd.DataFrame(fdr2).to_csv('heterogeneous_sc_fdr.csv')
pd.DataFrame(fm2).to_csv('heterogeneous_sc_fm.csv')

np.mean(fnp2, axis=0), np.std(fnp2, axis=0)
np.mean(fdr2, axis=0), np.std(fdr2, axis=0)
np.mean(fm2, axis=0), np.std(fm2, axis=0)

signal_est_jc_2 = np.zeros(100)
for i in range(0, 100):
    z_f = large_data2[0:128, 0:128, i].flatten()
    signal_est_jc_2[i] = epsest_func(z_f, 0, 1)
signal_est_jc_2 = signal_est_jc_2*128*128
ratio_jc_2 = signal_est_jc_2 / 1686
np.mean(ratio_jc_2, axis=0), np.std(ratio_jc_2, axis=0)
ratio2 = signal_est_post2/ 1686
np.mean(ratio2, axis=0), np.std(ratio2, axis=0)

### Heterogeneous 1
np.random.seed(2333)
random.seed(2333)
large_data5 = np.zeros((128, 128, 100))
large_signals5 = np.zeros((128, 128, 100))
for i in range(100):
    large_result5 = data_gen1(signal_prior, signal_weights1, signal_dist_well2)
    large_data5[0:128, 0:128, i] = large_result5['z']
    large_signals5[0:128, 0:128, i] = large_result5['flips']
fnp5 = np.zeros((100,11))
fdr5 = np.zeros((100,11))
fm5 = np.zeros((100,11))
signal_est5 = np.zeros(100)
signal_est_prior5 = np.zeros(100)
signal_est_post5 = np.zeros(100)
for i in range(25, 100):
    results_5 = simulation_run_new(large_data5[0:128, 0:128, i], large_signals5[0:128, 0:128, i], 0.05, 0.1)
    fnp5[i,:] = results_5['fnp']
    fdr5[i,:] = results_5['fdr']
    fm5[i,:] = results_5['fm']
    signal_est5[i] = results_5['signal_est']
    signal_est_prior5[i] = results_5['signal_est_prior']
    signal_est_post5[i] = results_5['signal_est_post']
pd.DataFrame(fnp5).to_csv('heterogeneous_sc_1_fnp.csv')
pd.DataFrame(fdr5).to_csv('heterogeneous_sc_1_fdr.csv')
pd.DataFrame(fm5).to_csv('heterogeneous_sc_1_fm.csv')

np.mean(fnp5, axis=0), np.std(fnp5, axis=0)
np.mean(fdr5, axis=0), np.std(fdr5, axis=0)
np.mean(fm5, axis=0), np.std(fm5, axis=0)

signal_est_jc_5 = np.zeros(100)
for i in range(0, 100):
    z_f = large_data5[0:128, 0:128, i].flatten()
    signal_est_jc_5[i] = epsest_func(z_f, 0, 1)
    
signal_est_jc_5 = signal_est_jc_5*128*128
ratio_jc_5 = signal_est_jc_5 / 1686
np.mean(ratio_jc_5, axis=0), np.std(ratio_jc_5, axis=0)
ratio5 = signal_est_post5 / 1686
np.mean(ratio5, axis=0), np.std(ratio5, axis=0)

## Heterogeneous 3
np.random.seed(2333)
random.seed(2333)
large_data6 = np.zeros((128, 128, 100))
large_signals6 = np.zeros((128, 128, 100))
for i in range(100):
    large_result6 = data_gen1(signal_prior, signal_weights1, signal_dist_well3)
    large_data6[0:128, 0:128, i] = large_result6['z']
    large_signals6[0:128, 0:128, i] = large_result6['flips']
fnp6 = np.zeros((100,11))
fdr6 = np.zeros((100,11))
fm6 = np.zeros((100,11))
signal_est6 = np.zeros(100)
signal_est_prior6 = np.zeros(100)
signal_est_post6 = np.zeros(100)
pd.DataFrame(fnp6).to_csv('heterogeneous_sc_3_fnp.csv')
pd.DataFrame(fdr6).to_csv('heterogeneous_sc_3_fdr.csv')
pd.DataFrame(fm6).to_csv('heterogeneous_sc_3_fm.csv')

np.mean(fnp6, axis=0), np.std(fnp6, axis=0)
np.mean(fdr6, axis=0), np.std(fdr6, axis=0)
np.mean(fm6, axis=0), np.std(fm6, axis=0)

signal_est_jc_6 = np.zeros(100)
for i in range(0, 100):
    z_f = large_data6[0:128, 0:128, i].flatten()
    signal_est_jc_6[i] = epsest_func(z_f, 0, 1)
    
signal_est_jc_6 = signal_est_jc_6*128*128
ratio_jc_6 = signal_est_jc_6 / 1686
np.mean(ratio_jc_6, axis=0), np.std(ratio_jc_6, axis=0)
ratio6 = signal_est_post6 / 1686
np.mean(ratio6, axis=0), np.std(ratio6, axis=0)


## Heterogeneous 2
np.random.seed(2333)
random.seed(2333)
large_data7 = np.zeros((128, 128, 100))
large_signals7 = np.zeros((128, 128, 100))
for i in range(100):
    large_result7 = data_gen1(signal_prior, signal_weights1, signal_dist_well4)
    large_data7[0:128, 0:128, i] = large_result7['z']
    large_signals7[0:128, 0:128, i] = large_result7['flips']
fnp7 = np.zeros((100,11))
fdr7 = np.zeros((100,11))
fm7 = np.zeros((100,11))
signal_est7 = np.zeros(100)
signal_est_prior7 = np.zeros(100)
signal_est_post7 = np.zeros(100)
pd.DataFrame(fnp7).to_csv('heterogeneous_sc_2_fnp.csv')
pd.DataFrame(fdr7).to_csv('heterogeneous_sc_2_fdr.csv')
pd.DataFrame(fm7).to_csv('heterogeneous_sc_2_fm.csv')

np.mean(fnp7, axis=0), np.std(fnp7, axis=0)
np.mean(fdr7, axis=0), np.std(fdr7, axis=0)
np.mean(fm7, axis=0), np.std(fm7, axis=0)

signal_est_jc_7 = np.zeros(100)
for i in range(0, 100):
    z_f = large_data7[0:128, 0:128, i].flatten()
    signal_est_jc_7[i] = epsest_func(z_f, 0, 1)
signal_est_jc_7 = signal_est_jc_7*128*128
ratio_jc_7 = signal_est_jc_7 / 1686
np.mean(ratio_jc_7, axis=0), np.std(ratio_jc_7, axis=0)
ratio7 = signal_est_post7 / 1686
np.mean(ratio7, axis=0), np.std(ratio7, axis=0)

## Correlated setting with strength 4 and 2
np.random.seed(2333)
random.seed(2333)
large_data3 = np.zeros((128, 128, 100))
large_signals3 = np.zeros((128, 128, 100))
for i in range(100):
    large_result3 = data_gen2(4, 2)
    large_data3[0:128, 0:128, i] = large_result3['z']
    large_signals3[0:128, 0:128, i] = large_result3['flips']
fnp3 = np.zeros((100,11))
fdr3 = np.zeros((100,11))
fm3 = np.zeros((100,11))
signal_est3 = np.zeros(100)
signal_est_prior3 = np.zeros(100)
signal_est_post3 = np.zeros(100)

for i in range(25, 100):
    results_3 = simulation_run_new(large_data3[0:128, 0:128, i], large_signals3[0:128, 0:128, i], 0.05, 0.1)
    fnp3[i,:] = results_3['fnp']
    fdr3[i,:] = results_3['fdr']
    fm3[i,:] = results_3['fm']
    signal_est3[i] = results_3['signal_est']
    signal_est_prior3[i] = results_3['signal_est_prior']
    signal_est_post3[i] = results_3['signal_est_post']

pd.DataFrame(fnp3).to_csv('new_42_fnp.csv')
pd.DataFrame(fdr3).to_csv('new_42_fdr.csv')
pd.DataFrame(fm3).to_csv('new_42_fm.csv')

np.mean(fnp3, axis=0), np.std(fnp3, axis=0)
np.mean(fdr3, axis=0), np.std(fdr3, axis=0)
np.mean(fm3, axis=0), np.std(fm3, axis=0)

signal_est_jc_3 = np.zeros(100)
for i in range(0, 100):
    z_f = large_data3[0:128, 0:128, i].flatten()
    signal_est_jc_3[i] = epsest_func(z_f, 0, 1)
    
signal_est_jc_3 = signal_est_jc_3*128*128
ratio_jc_3 = signal_est_jc_3 / 1125
np.mean(ratio_jc_3, axis=0), np.std(ratio_jc_3, axis=0)
ratio3 = signal_est_post3 / 1125
np.mean(ratio3, axis=0), np.std(ratio3, axis=0)

## Correlated setting with strength 2 and 1

np.random.seed(2333)
random.seed(2333)
large_data4 = np.zeros((128, 128, 100))
large_signals4 = np.zeros((128, 128, 100))
for i in range(100):
    large_result4 = data_gen2(2, 1)
    large_data4[0:128, 0:128, i] = large_result4['z']
    large_signals4[0:128, 0:128, i] = large_result4['flips']
fnp4 = np.zeros((100,11))
fdr4 = np.zeros((100,11))
fm4 = np.zeros((100,11))
signal_est4 = np.zeros(100)
signal_est_prior4 = np.zeros(100)
signal_est_post4 = np.zeros(100)
for i in range(16, 17):
    results_4 = simulation_run_new(large_data4[0:128, 0:128, i], large_signals4[0:128, 0:128, i], 0.05, 0.1)
    fnp4[i,:] = results_4['fnp']
    fdr4[i,:] = results_4['fdr']
    fm4[i,:] = results_4['fm']
    signal_est4[i] = results_4['signal_est']
    signal_est_prior4[i] = results_4['signal_est_prior']
    signal_est_post4[i] = results_4['signal_est_post']
pd.DataFrame(fnp4).to_csv('new_21_fnp.csv')
pd.DataFrame(fdr4).to_csv('new_21_fdr.csv')
pd.DataFrame(fm4).to_csv('new_21_fm.csv')

np.mean(fnp4, axis=0), np.std(fnp4, axis=0)
np.mean(fdr4, axis=0), np.std(fdr4, axis=0)
np.nanmean(fm4, axis=0), np.nanstd(fm4, axis=0)

signal_est_jc_4 = np.zeros(100)
for i in range(0, 100):
    z_f = large_data4[0:128, 0:128, i].flatten()
    signal_est_jc_4[i] = epsest_func(z_f, 0, 1)
    
signal_est_jc_4 = signal_est_jc_4*128*128
ratio_jc_4 = signal_est_jc_4 / 1125
np.mean(ratio_jc_4, axis=0), np.std(ratio_jc_4, axis=0)
ratio4 = signal_est_post4 / 1125
np.mean(ratio4, axis=0), np.std(ratio4, axis=0)

