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

To run a simple example, we can use the example data in `example/test_data.csv`. This is a simple 128 x 128 test dataset with two partially overlapped circle areas of signal. Running SMDR  on this is simple:

```python
import numpy as np
from smdr.main import smdr
data = np.loadtxt('example/test_data.csv', delimiter=',')

# Runs the SMDR screening algorithm with the different control levels
results1 = smdr(data, epsilon=0.1)
results2 = smdr(data, epsilon=0.05)
```

Visualizing of the results
------------------

Once you have run the algorithm, you can use the returned dictionary to analyze the results.

```python
import matplotlib.pylab as plt
fig, axs = plt.subplots(1,3, figsize=(15, 5))
axs[0].set_title('Raw data', fontsize=15)
axs[0].imshow(data, cmap='gray_r', vmin=0, vmax=1)
axs[1].set_title('SMDR(beta=0.1)', fontsize=15)
axs[1].imshow(results1['de'], cmap='gray_r', vmin=0, vmax=1)
axs[2].set_title('SMDR(beta=0.05)', fontsize=15)
axs[2].imshow(results2['de'], cmap='gray_r', vmin=0, vmax=1)
```
![Visualization the results](https://github.com/yifeihu93/smdr/blob/cf4fc2c65ff820e171a797d867004efb3ce203bb/example/test_results.png)
