import pygfl, sys, os, time, csv, math
import matplotlib.pylab as plt
import matplotlib.patches as patches
import numpy as np
import nibabel as nib
import scipy.stats as st
from utils import *
from collections import defaultdict
from collections import OrderedDict
from statsmodels.stats.multitest import local_fdr

# some of the following functions are revised from package smoothfdr
def edges_gen(data):
    ''' generate edges for given image data. '''
    np.set_printoptions(suppress=True)
    (x, y) = data.shape
    n = 2*x*y-x-y
    edge = np.zeros((n, 2))
    r = 0
    for i in range(x-1):
        for j in range(y-1):
            if (data[i, j]!=0 and data[i+1, j]!=0):
                edge[r, 0] = y*i+j
                edge[r, 1] = y*(i+1)+j
                r = r + 1
            if (data[i, j]!=0 and data[i, j+1]!=0):
                edge[r, 0] = y*i+j
                edge[r, 1] = y*i+j+1
                r = r + 1
        if (data[i, y-1]!=0 and data[i+1, y-1]!=0):
            edge[r, 0] = y*i+(y-1)
            edge[r, 1] = y*(i+1)+(y-1)
            r = r + 1
            print(i)
    for k in range(y-1):
        if (data[x-1, k]!=0 and data[x-1, k+1]!=0):
            edge[r, 0] = (x-1)*y+k
            edge[r, 1] = (x-1)*y+k+1
            r = r + 1
    edge = edge[~np.all(edge == 0, axis=1)]
    np.savetxt("edges_gen.csv", edge, delimiter=",", fmt='%d')
    return edge

def load_edges(filename):
''' load edges file from .cvs file. '''
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        edges = defaultdict(list)
        for line in reader:
            nodes = [int(x) for x in line]
            for n1,n2 in zip(nodes[:-1], nodes[1:]):
                edges[n1].append(n2)
                edges[n2].append(n1)
    return edges

def _local_agg_fdr_helper(fdr_level, p_star, ghat, ghat_lambda, wstar_lambda, tmin, tmax, tmin_fdr, tmax_fdr, rel_tol=1e-20):
    '''Finds the t-level via binary search.'''
    if np.isclose(tmin, tmax, atol=rel_tol) or np.isclose(tmin_fdr, tmax_fdr, atol=rel_tol) or tmax_fdr <= fdr_level:
        return (tmax, tmax_fdr) if tmax_fdr <= fdr_level else (tmin, tmin_fdr)
    tmid = (tmax + tmin) / 2.
    tmid_fdr = wstar_lambda * ghat(p_star, tmid) / (max(1,(p_star < tmid).sum()) * (1-ghat_lambda))
    print('t: [{0}, {1}, {2}] => fdr: [{3}, {4}, {5}]'.format(tmin, tmid, tmax, tmin_fdr, tmid_fdr, tmax_fdr))
    if tmid_fdr <= fdr_level:
        return _local_agg_fdr_helper(fdr_level, p_star, ghat, ghat_lambda, wstar_lambda, tmid, tmax, tmid_fdr, tmax_fdr)
    return _local_agg_fdr_helper(fdr_level, p_star, ghat, ghat_lambda, wstar_lambda, tmin, tmid, tmin_fdr, tmid_fdr)

def local_agg_fdr(pvals, edges, fdr_level, lmbda = 0.1):
    '''Given a list of p-values and the graph connecting them, applies a median
    filter to locally aggregate them and then performs a corrected FDR procedure
    from Zhang, Fan, and Yu (Annals of Statistics, 2011). lmbda is a tuning
    constant typically set to 0.1.'''
    p_star0 = median_filter(pvals, edges) # aggregate p-values
    p_star = p_star0[p_star0!=1]
    ghat = lambda p, t: (p >= (1-t)).sum() / max(1., (2.0 * (p > 0.5).sum() + (p==0.5).sum())) # empirical null CDF
    wstar_lambda = (p_star > lmbda).sum() # number of nonrejects at the level lambda
    ghat_lambda = ghat(p_star, lmbda) # empirical null CDF at rejection level lambda
    # Use binary search to find the highest t value that satisfies the fdr level
    tmin = 0.
    tmax = 1.
    tmin_fdr = wstar_lambda * ghat(p_star, tmin) / (max(1,(p_star < tmin).sum()) * (1-ghat_lambda))
    tmax_fdr = wstar_lambda * ghat(p_star, tmax) / (max(1,(p_star < tmax).sum()) * (1-ghat_lambda))
    t, tfdr = _local_agg_fdr_helper(fdr_level, p_star, ghat, ghat_lambda, wstar_lambda, tmin, tmax, tmin_fdr, tmax_fdr)
    print('t: {0} tfdr: {1}'.format(t, tfdr))
    # Returns the indices of all discoveries
    return np.where(p_star0 < t)[0]

def fdrl_run(data, edges, fdr_level):
    ''' wrapper function of FDRL procedure. '''
    data = data
    p_values = p_value(data, 0, 1)
    #p_values = 1.0 - st.norm.cdf(data)
    p_values_flat = p_values.flatten()
    dis_fdrl = local_agg_fdr(p_values_flat, edges, fdr_level, lmbda = 0.2)
    results_fdrl = np.zeros(p_values_flat.shape)
    results_fdrl[dis_fdrl] = 1
    results_fdrl = results_fdrl.reshape(data.shape)
    return(results_fdrl)

def benjamini_hochberg(z, fdr, mu0=0., sigma0=1.):
    '''Performs Benjamini-Hochberg multiple hypothesis testing on z at the given false discovery rate threshold.'''
    z_shape = z.shape if len(z.shape) > 1 else None
    if z_shape is not None:
        z = z.flatten()
    #p = p_value(z, mu0=mu0, sigma0=sigma0)
    p = 1.0 - st.norm.cdf(z)
    p_orders = np.argsort(p)
    discoveries = []
    m = float(len(p_orders))
    for k, s in enumerate(p_orders):
        if p[s] <= (k+1) / m * fdr:
            discoveries.append(s)
        else:
            break
    discoveries = np.array(discoveries)
    if z_shape is not None:
        x = np.zeros(z.shape)
        x[discoveries] = 1
        discoveries = np.where(x.reshape(z_shape) == 1)
    return discoveries

def bh(data, fdr_level):
    ''' wrapper function of BH-FDR procedure. '''
    dis_bh = benjamini_hochberg(data, fdr_level, mu0=0., sigma0=1.)
    results_bh = np.zeros(data.shape)
    results_bh[dis_bh] = 1
    return(results_bh)


def epsest_func(x, u, sigma):
    ''' signal proportion estimator proposed by Meinshausen, N., J. Rice, et al. (2006)'''
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

def mdr(z0, epsilon, deg, nbins):
    ''' MDR screening procedure proposed by Tony Cai, T. and W. Sun (2017). '''
    z = z0[z0!=0]
    z_f = z.flatten()
    lfdr = local_fdr(z_f, deg=deg, nbins=nbins)
    eps = epsest_func(z_f, 0, 1)
    n = lfdr.size
    de = np.zeros(n)
    nmp = n*eps*epsilon
    tor = 1-lfdr
    tor.sort()
    cumsum_tor = np.cumsum(tor)
    k = np.sum(cumsum_tor <= nmp)
    threshold = 1-tor[k-1]
    reject = (lfdr<=threshold)    
    t = np.min(z_f[np.logical_and(reject == 1, z_f > 0)])
    de = np.zeros(z0.shape)
    de[np.where(z0>t)] = 1
    return {'th': t, 'de': de}

def afnc(z0, epsilon):
    ''' AFNC procedure proposed by
    Jeng, X. J., Z. J. Daye, W. Lu, and J.-Y. Tzeng (2016). '''
    z = z0[z0!=0]
    n = z.size
    z_f = z.flatten()
    s = int(epsest_func(z_f, 0, 1) * n)
    #pval = p_value(z_f)
    pval = 1.0 - st.norm.cdf(z_f)
    #p_order = p_value(z_f)
    p_order = 1.0 - st.norm.cdf(z_f)
    p_order.sort()
    p_order_s = p_order[s:n-1]
    numNul = p_order_s.size
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
        jstar = s + jhat
    t = p_order[jstar-1]
    pval0 = 1.0 - st.norm.cdf(z0)
    #pval0 = p_value(z0)
    reject = (pval0<=t)
    de = np.zeros(z0.shape)
    de[reject] = 1
    return(de)

def smdr(z, epsilon, verbose=5, missing_val=0):
    ''' Smooth MDR screening procedure proposed by our research. '''
    results_fdrs = smooth_fdr(z, 0.05, verbose=verbose, missing_val=missing_val)
    post = results_fdrs['posteriors']
    lfdr0 = 1-results_fdrs['posteriors']
    n0 = lfdr0.size
    de = np.zeros(z.shape)
    n = np.sum(z!=0)
    eps = np.sum(post)/n
    nmp = n*eps*epsilon
    lfdr_f = lfdr0[np.where(z!=0)].flatten()
    tor = 1-lfdr_f
    tor.sort()
    cumsum_tor = np.cumsum(tor)
    k = np.sum(cumsum_tor <= nmp)
    threshold = 1-tor[k-1]
    reject = (lfdr0<=threshold)
    accept = (lfdr0>threshold)
    de[np.logical_and(reject == 1, image_data_slice >0 )] = 1
    return {'nr': k,
			'th': threshold,
			'de': de}

