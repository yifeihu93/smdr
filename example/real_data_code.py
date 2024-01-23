# Import package
import pygfl, sys, os, time, ramdom, matplotlib, math
import matplotlib.pylab as plt
import numpy as np
import nibabel as nib
import scipy.stats as st
from random import sample
from collections import defaultdict, OrderedDict
from statsmodels.stats.multitest import local_fdr
from smoothfdr import smooth_fdr, local_agg_fdr

# Load image data
image_data_slice = np.loadtxt('Desktop/simulation_test/fmri_slice_zscores.csv', delimiter=',', skiprows=1)

# FDRL
def fdrl_run(data, edges, fdr_level):
    data = data
    p_values = p_value(data, 0, 1)
    #p_values = 1.0 - st.norm.cdf(data)
    p_values_flat = p_values.flatten()
    dis_fdrl = local_agg_fdr(p_values_flat, edges, fdr_level, lmbda = 0.2)
    results_fdrl = np.zeros(p_values_flat.shape)
    results_fdrl[dis_fdrl] = 1
    results_fdrl = results_fdrl.reshape(data.shape)
    return(results_fdrl)

def edges_gen(data):
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
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        edges = defaultdict(list)
        for line in reader:
            nodes = [int(x) for x in line]
            for n1,n2 in zip(nodes[:-1], nodes[1:]):
                edges[n1].append(n2)
                edges[n2].append(n1)
    return edges

np.random.seed(2333)
start_fdrl = time.time()
de_fdrl = np.logical_and(fdrl_run(image_data_slice, edges_prun, 0.1) == 1, image_data_slice>0 )
end_fdrl = time.time()
time_fdrl = end_fdrl - start_fdrl
plt.imshow(de_fdrl, cmap='gray_r', vmin=0, vmax=1)

# FDRS
edges = load_edges('Desktop/simulation_test/edges.csv')
np.random.seed(2333)
results_fdrs = smooth_fdr(image_data_slice, 0.1, verbose=5, missing_val=0)
de_fdrs = np.zeros(image_data_slice.shape)
de_fdrs[np.logical_and(results_fdrs['discoveries'] == 1, image_data_slice >0 )] = 1
de_fdrs.sum()
plt.imshow(de_fdrs, cmap='gray_r', vmin=0, vmax=1)

# BH-FDR
def benjamini_hochberg(z, fdr, mu0=0., sigma0=1.):
    z_shape = z.shape if len(z.shape) > 1 else None
    if z_shape is not None:
        z = z.flatten()
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

def bh_run(data, fdr_level):
    dis_bh = benjamini_hochberg(data, fdr_level, mu0=0., sigma0=1.)
    results_bh = np.zeros(data.shape)
    results_bh[dis_bh] = 1
    return(results_bh)

de_bh = bh_run(image_data_slice, 0.1)
plt.imshow(de_bh, cmap='gray_r', vmin=0, vmax=1)

# MDR
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

def MDR_run(z0, epsilon, deg, nbins):
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

    return {'th': t,
			'de': de}

np.random.seed(2333)
de_mdr = MDR_run(image_data_slice, 0.1, 7, 30)
de_mdr, de_mdr['de'].sum()
plt.imshow(de_mdr['de'], cmap='gray_r', vmin=0, vmax=1)

# AFNC
import scipy.stats as st
def AFNC_run(z0, epsilon):
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

np.random.seed(2333)
start_afnc = time.time()
de_afnc = AFNC_run(image_data_slice, 0.1)
plt.imshow(de_afnc, cmap='gray_r', vmin=0, vmax=1)

# SMDR
def smdr_proc(z, epsilon):
    results_fdrs = smooth_fdr(image_data_slice, 0.05, verbose=5, missing_val=0)
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

de_smdr1 = smdr_proc(results_fdrs['posteriors'], image_data_slice, 1-results_fdrs['posteriors'], 0.1)
plt.imshow(de_smdr1['de'], cmap='gray_r', vmin=0, vmax=1)

de_smdr2 = smdr_proc(results_fdrs['posteriors'], image_data_slice, 1-results_fdrs['posteriors'], 0.05)
plt.imshow(de_smdr2['de'], cmap='gray_r', vmin=0, vmax=1)


# Plot
## Compare different methods
import matplotlib.pyplot as plt

my_red_cmap = plt.cm.Reds
my_red_cmap.set_under(color="white", alpha="0.3")

lineWidth = 30
plt.figure()

plt.subplot(4, 2, 1)
plt.imshow(image_data_slice, cmap='gray_r', vmin=0, vmax=1)
plt.title('Z statistics', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.subplot(4, 2, 2)
plt.imshow(image_data_slice, cmap='gray_r')
plt.imshow(de_bh, cmap=my_red_cmap, alpha = 0.5, vmin=0, vmax=1)
plt.title('BH-FDR(alpha=0.1)', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.subplot(4, 2, 3)
plt.imshow(image_data_slice, cmap='gray_r')
plt.imshow(de_fdrl, cmap=my_red_cmap, alpha = 0.5, vmin=0, vmax=1)
plt.title('FDRL(alpha=0.1)', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.subplot(4, 2, 4)
plt.imshow(image_data_slice, cmap='gray_r')
plt.imshow(de_fdrs, cmap=my_red_cmap, alpha = 0.5, vmin=0, vmax=1)
plt.title('FDRS(alpha=0.1)', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.subplot(4, 2, 5)
plt.imshow(image_data_slice, cmap='gray_r')
plt.imshow(de_mdr['de'], cmap=my_red_cmap, alpha = 0.5, vmin=0, vmax=1)
plt.title('CS-MDR(beta=0.1)', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.subplot(4, 2, 6)
plt.imshow(image_data_slice, cmap='gray_r')
plt.imshow(de_afnc, cmap=my_red_cmap, alpha = 0.5, vmin=0, vmax=1)
plt.title('AFNC(beta=0.1)', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.subplot(4, 2, 7)
plt.imshow(image_data_slice, cmap='gray_r')
plt.imshow(de_smdr1['de'], cmap=my_red_cmap, alpha = 0.5, vmin=0, vmax=1)
plt.title('SMDR(beta=0.1)', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.subplot(4, 2, 8)
plt.imshow(image_data_slice, cmap='gray_r')
plt.imshow(de_smdr2['de'], cmap=my_red_cmap, alpha = 0.5, vmin=0, vmax=1)
plt.title('SMDR(beta=0.05)', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=14)


## Compare FDRS and SMDR (0.1)
layer1 = de_fdrs[20:110, 0:105]
layer2 = de_smdr1['de'][20:110, 0:105]-de_fdrs[20:110, 0:105]
image_data_slice[np.where(image_data_slice!=0)] = 1
layer0 = image_data_slice[20:110, 0:105]

new_img = 1*layer1 + 3*layer2 + 1*layer0
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.annotate("BA7",xy=(20,50),xytext=(+30,-40),textcoords='offset points',fontsize=40,
             arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.1'))
plt.annotate("BA6",xy=(80,65),xytext=(+30,-40),textcoords='offset points',fontsize=40,
             arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.1'))
ax = plt.gca()
rect1 = patches.Rectangle((5,45),
                 15,
                 12,
                 linewidth=2,
                 edgecolor='black',
                 fill = False)
ax.add_patch(rect1)
rect2 = patches.Rectangle((70,57),
                 10,
                 10,
                 linewidth=2,
                 edgecolor='black',
                 fill = False)
ax.add_patch(rect2)
plt.imshow(new_img, cmap='Greys', vmin=0, vmax=5)


## Compare FDRS and SMDR (0.05)
layer1 = de_fdrs[20:110, 0:105]
layer2 = de_smdr2['de'][20:110, 0:105]-de_fdrs[20:110, 0:105]
image_data_slice[np.where(image_data_slice!=0)] = 1
layer0 = image_data_slice[20:110, 0:105]

new_img = 1*layer1 + 3*layer2 + 1*layer0
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.annotate("BA7",xy=(20,50),xytext=(+30,-40),textcoords='offset points',fontsize=40,
             arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.1'))
plt.annotate("BA6",xy=(80,65),xytext=(+30,-40),textcoords='offset points',fontsize=40,
             arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.1'))
ax = plt.gca()
rect1 = patches.Rectangle((5,45),
                 15,
                 12,
                 linewidth=2,
                 edgecolor='black',
                 fill = False)
ax.add_patch(rect1)
rect2 = patches.Rectangle((70,57),
                 10,
                 10,
                 linewidth=2,
                 edgecolor='black',
                 fill = False)
ax.add_patch(rect2)
plt.imshow(new_img, cmap='Greys', vmin=0, vmax=5)
