Spatially Adaptive MDR Screening (smdr)
==========================================


The `smdr` package provides a novel spatially adaptive variable screening procedure to
enable effective control of false negatives while leveraging the spatial structure of fMRI data.
Compared to existing methods built upon multiple hypothesis testing, the new procedure is less
stringent in false positive control to trade for better identification and protection for functional
regions. The new method is also substantially different from existing false negative control
procedures which do not take spatial information into account.

Installation
------------

To install the Python version:

```
pip install smdr
```

You can then run the tool directly from the terminal by typing the `smdr` command. If you want to integrate it into your code, you can simply `import smdr`.



Running an example
------------------

There are lots of parameters that you can play with if you so choose, but one of the biggest benefit of FDR smoothing is that you don't have to worry about it in most cases.

To run a simple example, we can use the example data in `example/data.csv`. This is a simple 128x128 test dataset with two plateaus of increased prior probability of signal. Running FDR smoothing on this is simple:

```python
import numpy as np
from smdr.main import smdr

data = np.loadtxt('example/test_data.csv', delimiter=',')
epsilon = 0.1

# Runs the SMDR screening algorithm with the default settings
results = smdr(data, epsilon=epsilon)
```

Visualizing of the results
------------------

To run a simple example, we can use the example data in `example/data.csv`. This is a simple 128x128 test dataset with two plateaus of increased prior probability of signal. Running SMDR smoothing on this is simple:

```python
import matplotlib.pylab as plt
fig, ax = plt.subplots(1,2)
ax[0,0].imshow(data, cmap='gray_r')
ax[0,0].set_title('Raw data')
ax[0,1].imshow(results['de'], cmap='gray_r', vmin=0, vmax=1)
ax[0,1].set_title('SMDR')
```
![Visualization the results](https://raw.githubusercontent.com/yifehu93/smdr/example/test_results.png)
