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



image_data_slice = np.loadtxt('example/fmri_slice_zscores.csv', delimiter=',', skiprows=1)
plt.plot(sorted(image_data_slice.flatten(),reverse = True))
edges = load_edges('example/edges.csv')

edges_gen(image_data_slice)
edges_prun = load_edges('edges_gen.csv')
# FDRL
np.random.seed(2333)
start_fdrl = time.time()
de_fdrl = np.logical_and(fdrl(image_data_slice, edges_prun, 0.05) == 1, image_data_slice>0 )
end_fdrl = time.time()
time_fdrl = end_fdrl - start_fdrl
plt.imshow(de_fdrl, cmap='gray_r', vmin=0, vmax=1)

de_fdrl.sum()

time_fdrl


# FDRS
np.random.seed(2333)
time_start=time.time()
results_fdrs = smooth_fdr(image_data_slice, 0.05, verbose=5, missing_val=0)
time_end=time.time()
print('time cost',time_end-time_start,'s')

de_fdrs = np.zeros(image_data_slice.shape)
de_fdrs[np.logical_and(results_fdrs['discoveries'] == 1, image_data_slice >0 )] = 1
de_fdrs.sum()

plt.imshow(de_fdrs, cmap='gray_r', vmin=0, vmax=1)

de_fdrs.sum()


# # BH-FDR
start_bh = time.time()
de_bh = bh(image_data_slice, 0.05)
end_bh = time.time()
plt.imshow(de_bh, cmap='gray_r', vmin=0, vmax=1)
print(end_bh-start_bh)

de_bh.sum()


# # MDR
np.random.seed(2333)
start_mdr = time.time()
de_mdr = mdr(image_data_slice, 0.1, 7, 30)
end_mdr = time.time()
de_mdr, de_mdr['de'].sum()

print(end_mdr - start_mdr)

plt.imshow(de_mdr['de'], cmap='gray_r', vmin=0, vmax=1)


# # AFNC
np.random.seed(2333)
start_afnc = time.time()
de_afnc = afnc(image_data_slice, 0.1)
end_afnc = time.time()
de_afnc, de_afnc.sum()

print(end_afnc - start_afnc)

plt.imshow(de_afnc, cmap='gray_r', vmin=0, vmax=1)



# # SMDR

de_smdr01 = smdr(image_data_slice, 0.1)
np.sum(de_smdr01['de']!=0)

plt.imshow(de_new2['de'], cmap='gray_r', vmin=0, vmax=1)


de_smdr02 = smdr(image_data_slice, 0.2)
np.sum(smdr02['de']!=0)

plt.imshow(de_smdr02['de'], cmap='gray_r', vmin=0, vmax=1)


de_smdr005 = smdr(image_data_slice, 0.05)
np.sum(smdr02['de']!=0)

plt.imshow(de_new4['de'], cmap='gray_r', vmin=0, vmax=1)


start_smdr = time.time()
results_smdr = smdr_proc(image_data_slice, 0.05)
end_smdr = time.time()
print(end_smdr - start_smdr)


# # Plot

plt.rcParams["figure.figsize"] = (20,40)
fig, ax = plt.subplots(4,2)

ax[0,0].imshow(image_data_slice, cmap='gray_r', vmin=0, vmax=1)
ax[0,0].set_title('Raw', fontsize=25)
ax[0,0].tick_params(axis='both', which='major', labelsize=14)

ax[0,1].imshow(de_bh, cmap='gray_r', vmin=0, vmax=1)
ax[0,1].set_title('BH-FDR', fontsize=25)
ax[0,1].tick_params(axis='both', which='major', labelsize=14)

ax[1,0].imshow(de_fdrl, cmap='gray_r', vmin=0, vmax=1)
ax[1,0].set_title('FDRL', fontsize=25)
ax[1,0].tick_params(axis='both', which='major', labelsize=14)

ax[1,1].imshow(de_fdrs, cmap='gray_r', vmin=0, vmax=1)
ax[1,1].set_title('FDRS', fontsize=25)
ax[1,1].tick_params(axis='both', which='major', labelsize=14)

ax[2,0].imshow(de_mdr['de'], cmap='gray_r', vmin=0, vmax=1)
ax[2,0].set_title('MDR', fontsize=25)
ax[2,0].tick_params(axis='both', which='major', labelsize=14)

ax[2,1].imshow(de_afnc, cmap='gray_r', vmin=0, vmax=1)
ax[2,1].set_title('AFNC', fontsize=25)
ax[2,1].tick_params(axis='both', which='major', labelsize=14)

ax[3,0].imshow(de_smdr01['de'], cmap='gray_r', vmin=0, vmax=1)
ax[3,0].set_title('SMDR(beta=0.1)', fontsize=25)
ax[3,0].tick_params(axis='both', which='major', labelsize=14)

ax[3,1].imshow(de_smdr005['de'], cmap='gray_r', vmin=0, vmax=1)
ax[3,1].set_title('SMDR(beta=0.05)', fontsize=25)
ax[3,1].tick_params(axis='both', which='major', labelsize=14)




import matplotlib.pyplot as plt

my_red_cmap = plt.cm.Reds
my_red_cmap.set_under(color="white", alpha="0.3")

lineWidth = 30
plt.figure()

plt.subplot(4, 2, 1)
plt.imshow(image_data_slice, cmap='gray_r', vmin=0, vmax=1)
plt.title('Raw', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.subplot(4, 2, 2)
plt.imshow(image_data_slice, cmap='gray_r')
plt.imshow(de_bh, cmap=my_red_cmap, alpha = 0.5, vmin=0, vmax=1)
plt.title('BH-FDR', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.subplot(4, 2, 3)
plt.imshow(image_data_slice, cmap='gray_r')
plt.imshow(de_fdrl, cmap=my_red_cmap, alpha = 0.5, vmin=0, vmax=1)
plt.title('FDRL', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.subplot(4, 2, 4)
plt.imshow(image_data_slice, cmap='gray_r')
plt.imshow(de_fdrs, cmap=my_red_cmap, alpha = 0.5, vmin=0, vmax=1)
plt.title('FDRS', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.subplot(4, 2, 5)
plt.imshow(image_data_slice, cmap='gray_r')
plt.imshow(de_mdr['de'], cmap=my_red_cmap, alpha = 0.5, vmin=0, vmax=1)
plt.title('MDR', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.subplot(4, 2, 6)
plt.imshow(image_data_slice, cmap='gray_r')
plt.imshow(de_afnc, cmap=my_red_cmap, alpha = 0.5, vmin=0, vmax=1)
plt.title('AFNC', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.subplot(4, 2, 7)
plt.imshow(image_data_slice, cmap='gray_r')
plt.imshow(de_smdr01['de'], cmap=my_red_cmap, alpha = 0.5, vmin=0, vmax=1)
plt.title('SMDR(beta=0.1)', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.subplot(4, 2, 8)
plt.imshow(image_data_slice, cmap='gray_r')
plt.imshow(de_smdr005['de'], cmap=my_red_cmap, alpha = 0.5, vmin=0, vmax=1)
plt.title('SMDR(beta=0.05)', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=14)



from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# create dummy data
zvals = image_data_slice# np.random.rand(100,100)*10-5
zvals2 = de_bh

# generate the colors for your colormap
color1 = colorConverter.to_rgba('white')
color2 = colorConverter.to_rgba('black')

# make the colormaps
cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['white','black'],256)
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)

cmap2._init() # create the _lut array, with rgba values

# create your alpha array and fill the colormap with them.
# here it is progressive, but you can create whathever you want
alphas = np.linspace(0, 0.8, cmap2.N+3)
cmap2._lut[:,-1] = alphas

img2 = plt.imshow(zvals, interpolation='nearest', cmap=cmap1, origin='lower')
img3 = plt.imshow(zvals2, interpolation='nearest', cmap=cmap2, origin='lower')

plt.show()




new_img = 2*de_new2['de']-de_fdrs
new_img[np.where(new_img == 2)] = -1
new_img[np.where(new_img == 1)] = 2
plt.imshow(new_img, cmap='RdGy', vmin=-2, vmax=2)




layer1 = de_fdrs[20:110, 0:105]
layer2 = de_smdr01['de'][20:110, 0:105]-de_fdrs[20:110, 0:105]
image_data_slice[np.where(image_data_slice!=0)] = 1
layer0 = image_data_slice[20:110, 0:105]

new_img = 1*layer1 + 3*layer2 + 1*layer0
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.annotate("BA7",xy=(20,50),xytext=(+30,-40),textcoords='offset points',fontsize=25,
             arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.1'))
plt.annotate("BA6",xy=(80,65),xytext=(+30,-40),textcoords='offset points',fontsize=25,
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



layer1 = de_fdrs[20:110, 0:105]
layer2 = de_smdr005['de'][20:110, 0:105]-de_fdrs[20:110, 0:105]
image_data_slice[np.where(image_data_slice!=0)] = 1
layer0 = image_data_slice[20:110, 0:105]

new_img = 1*layer1 + 3*layer2 + 1*layer0
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.annotate("BA7",xy=(20,50),xytext=(+30,-40),textcoords='offset points',fontsize=25,
             arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.1'))
plt.annotate("BA6",xy=(80,65),xytext=(+30,-40),textcoords='offset points',fontsize=25,
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

