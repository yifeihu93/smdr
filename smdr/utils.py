import numpy as np
import csv
from collections import defaultdict
from scipy.sparse import csc_matrix, lil_matrix
import scipy.stats as st

class ProxyDistribution:
    '''Simple proxy distribution to enable specifying signal distributions from the command-line'''
    def __init__(self, name, pdf_method, sample_method):
        self.name = name
        self.pdf_method = pdf_method
        self.sample_method = sample_method

    def pdf(self, x):
        return self.pdf_method(x)

    def sample(self, count=1):
        if count == 1:
            return self.sample_method()
        return np.array([self.sample_method() for _ in range(count)])

    def __repr__(self):
        return self.name

def generate_data_helper(flips, null_mean, null_stdev, signal_dist):
    '''Recursively builds multi-dimensional datasets.'''
    if len(flips.shape) > 1:
        return np.array([generate_data_helper(row, null_mean, null_stdev, signal_dist) for row in flips])

    # If we're on the last dimension, return the vector
    return np.array([signal_dist.sample() if flip else 0 for flip in flips]) + np.random.normal(loc=null_mean, scale=null_stdev, size=len(flips))

def generate_data(null_mean, null_stdev, signal_dist, signal_weights):
    '''Create a synthetic dataset.'''
    # Flip biased coins to decide which distribution to draw each sample from
    flips = np.random.random(size=signal_weights.shape) < signal_weights

    # Recursively generate the dataset
    samples = generate_data_helper(flips, null_mean, null_stdev, signal_dist)

    # Observed z-scores
    z = (samples - null_mean) / null_stdev

    return (z, flips)

def calc_fdr(probs, fdr_level):
    '''Calculates the detected signals at a specific false discovery rate given the posterior probabilities of each point.'''
    pshape = probs.shape
    if len(probs.shape) > 1:
        probs = probs.flatten()
    post_orders = np.argsort(probs)[::-1]
    avg_fdr = 0.0
    end_fdr = 0
    
    for idx in post_orders:
        test_fdr = (avg_fdr * end_fdr + (1.0 - probs[idx])) / (end_fdr + 1.0)
        if test_fdr > fdr_level:
            break
        avg_fdr = test_fdr
        end_fdr += 1

    is_finding = np.zeros(probs.shape, dtype=int)
    is_finding[post_orders[0:end_fdr]] = 1
    if len(pshape) > 1:
        is_finding = is_finding.reshape(pshape)
    return is_finding

def filter_nonrectangular_data(data, filter_value=0):
    '''Convert the square matrix to a vector containing only the values different than the filter values.'''
    x = data != filter_value
    nonrect_vals = np.arange(x.sum())
    nonrect_to_data = np.zeros(data.shape, dtype=int) - 1
    data_to_nonrect = np.where(x.T)
    data_to_nonrect = (data_to_nonrect[1],data_to_nonrect[0])
    nonrect_to_data[data_to_nonrect] = nonrect_vals
    nonrect_data = data[x]
    return (nonrect_data, nonrect_to_data, data_to_nonrect)

def sparse_2d_penalty_matrix(data_shape, nonrect_to_data=None):
    '''Create a sparse 2-d penalty matrix. Optionally takes a map to corrected indices, useful when dealing with non-rectangular data.'''
    row_counter = 0
    data = []
    row = []
    col = []

    if nonrect_to_data is not None:
        for j in range(data_shape[1]):
            for i in range(data_shape[0]-1):
                idx1 = nonrect_to_data[i,j]
                idx2 = nonrect_to_data[i+1,j]
                if idx1 < 0 or idx2 < 0:
                    continue
                row.append(row_counter)
                col.append(idx1)
                data.append(-1)

                row.append(row_counter)
                col.append(idx2)
                data.append(1.)

                row_counter += 1
        for j in range(data_shape[1]-1):
            for i in range(data_shape[0]):
                idx1 = nonrect_to_data[i,j]
                idx2 = nonrect_to_data[i,j+1]
                if idx1 < 0 or idx2 < 0:
                    continue
                row.append(row_counter)
                col.append(idx1)
                data.append(-1)

                row.append(row_counter)
                col.append(idx2)
                data.append(1.)

                row_counter += 1
    else:
        for j in range(data_shape[1]):
            for i in range(data_shape[0] - 1):
                idx1 = i+j*data_shape[0]
                idx2 = i+j*data_shape[0]+1

                row.append(row_counter)
                col.append(idx1)
                data.append(-1.)

                row.append(row_counter)
                col.append(idx2)
                data.append(1.)

                row_counter += 1

        col_counter = 0
        for i in range(data_shape[0]):
            for j in range(data_shape[1] - 1):
                idx1 = col_counter
                idx2 = col_counter+data_shape[0]

                row.append(row_counter)
                col.append(idx1)
                data.append(-1.)

                row.append(row_counter)
                col.append(idx2)
                data.append(1.)

                row_counter += 1
                col_counter += 1

    num_rows = row_counter
    num_cols = max(col) + 1
    return csc_matrix((data, (row, col)), shape=(num_rows, num_cols))
    
def sparse_1d_penalty_matrix(data_len):
    penalties = np.eye(data_len, dtype=float)[0:-1] * -1
    for i in range(len(penalties)):
        penalties[i,i+1] = 1
    return csc_matrix(penalties)

def cube_trails(xmax, ymax, zmax):
    '''Produces a list of trails following a simple row/col/aisle split strategy for a cube.'''
    trails = []
    for x in range(xmax):
        for y in range(ymax):
            trails.append([x * ymax * zmax + y * zmax + z for z in range(zmax)])
    for y in range(ymax):
        for z in range(zmax):
            trails.append([x * ymax * zmax + y * zmax + z for x in range(xmax)])
    for z in range(zmax):
        for x in range(xmax):
            trails.append([x * ymax * zmax + y * zmax + z for y in range(ymax)])
    return trails

def val_present(data, x, missing_val):
    return missing_val is None or x

def cube_edges(data, missing_val=None):
    '''Produces a list of edges for a cube with potentially missing data.
    If missing_val is specified, entries with that value will be considered
    missing and no edges will be connected to them.'''
    edges = []
    xmax, ymax, zmax = data.shape
    for y in range(ymax):
        for z in range(zmax):
            edges.extend([((x1, y, z), (x2, y, z))
                            for x1, x2 in zip(range(data.shape[0]-1), range(1,data.shape[0]))
                            if missing_val is None or (data[x1,y,z] != missing_val and data[x2,y,z] != missing_val)])
    for x in range(xmax):
        for z in range(zmax):
            edges.extend([((x, y1, z), (x, y2, z))
                            for y1, y2 in zip(range(data.shape[1]-1), range(1,data.shape[1]))
                            if missing_val is None or (data[x,y1,z] != missing_val and data[x,y2,z] != missing_val)])
    for x in range(xmax):
        for y in range(ymax):
            edges.extend([((x, y, z1), (x, y, z2))
                            for z1, z2 in zip(range(data.shape[2]-1), range(1,data.shape[2]))
                            if missing_val is None or (data[x,y,z1] != missing_val and data[x,y,z2] != missing_val)])
    return edges

def cube_trails_missing_helper(data, trails, cur_trail, v1, v2, missing_val):
    if data[v1] == missing_val or data[v2] == missing_val:
        if len(cur_trail) > 1:
            trails.append(cur_trail)
            cur_trail = []
    else:
        if len(cur_trail) == 0:
            cur_trail.append(v1)
        cur_trail.append(v2)
    return cur_trail

def cube_trails_missing(data, missing_val=None):
    '''Generates row/col/aisle trails for a cube when there may be missing data.'''
    trails = []
    xmax, ymax, zmax = data.shape
    for y in range(ymax):
        for z in range(zmax):
            cur_trail = []
            for x1, x2 in zip(range(data.shape[0]-1), range(1,data.shape[0])):
                v1 = (x1,y,z)
                v2 = (x2,y,z)
                cur_trail = cube_trails_missing_helper(data, trails, cur_trail, v1, v2, missing_val)
            if len(cur_trail) > 1:
                trails.append(cur_trail)
                
    for x in range(xmax):
        for z in range(zmax):
            cur_trail = []
            for y1, y2 in zip(range(data.shape[1]-1), range(1,data.shape[1])):
                v1 = (x,y1,z)
                v2 = (x,y2,z)
                cur_trail = cube_trails_missing_helper(data, trails, cur_trail, v1, v2, missing_val)
            if len(cur_trail) > 1:
                trails.append(cur_trail)

    for x in range(xmax):
        for y in range(ymax):
            cur_trail = []
            for z1, z2 in zip(range(data.shape[2]-1), range(1,data.shape[2])):
                v1 = (x, y, z1)
                v2 = (x, y, z2)
                cur_trail = cube_trails_missing_helper(data, trails, cur_trail, v1, v2, missing_val)
            if len(cur_trail) > 1:
                trails.append(cur_trail)
                            
    return trails


def load_trails(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        return load_trails_from_reader(reader)

def load_trails_from_reader(reader):
    trails = []
    breakpoints = []
    edges = defaultdict(list)
    for line in reader:
        if len(trails) > 0:
            breakpoints.append(len(trails))
        nodes = [int(x) for x in line]
        trails.extend(nodes)
        for n1,n2 in zip(nodes[:-1], nodes[1:]):
            edges[n1].append(n2)
            edges[n2].append(n1)
    if len(trails) > 0:
        breakpoints.append(len(trails))
    return (len(breakpoints), np.array(trails, dtype="int32"), np.array(breakpoints, dtype="int32"), edges)

def save_trails(filename, trails):
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(trails)

def pretty_str(p, decimal_places=2):
    '''Pretty-print a matrix or vector.'''
    if len(p.shape) == 1:
        return vector_str(p, decimal_places)
    if len(p.shape) == 2:
        return matrix_str(p, decimal_places)
    raise Exception('Invalid array with shape {0}'.format(p.shape))

def matrix_str(p, decimal_places=2):
    '''Pretty-print the matrix.'''
    return '[{0}]'.format("\n  ".join([vector_str(a, decimal_places) for a in p]))

def vector_str(p, decimal_places=2):
    '''Pretty-print the vector values.'''
    style = '{0:.' + str(decimal_places) + 'f}'
    return '[{0}]'.format(", ".join([style.format(a) for a in p]))

def mean_filter(pvals, edges, rescale=True):
    '''Given a list of p-values and their neighbors, applies a mean filter
    that replaces each p_i with p*_i where p*_i = mean(neighbors(p_i)).
    If rescale is true, then the p-values are rescaled to be variance 1.'''
    return np.array([np.mean(pvals[edges[i] + [i]]) * (np.sqrt(len(edges[i]) + 1) if rescale else 1) for i,p in enumerate(pvals)])

def median_filter(pvals, edges):
    '''Given a list of p-values and their neighbors, applies a median filter
    that replaces each p_i with p*_i where p*_i = median(neighbors(p_i)).'''
    return np.array([np.median(pvals[edges[i] + [i]]) for i,p in enumerate(pvals)])

def _local_agg_fdr_helper(fdr_level, p_star, ghat, ghat_lambda, wstar_lambda, tmin, tmax, tmin_fdr, tmax_fdr, rel_tol=1e-4):
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
    p_star = median_filter(pvals, edges) # aggregate p-values
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
    return np.where(p_star < t)[0]

def p_value(z, mu0=0., sigma0=1.):
    return 2*(1.0 - st.norm.cdf(np.abs((z - mu0) / sigma0)))

def benjamini_hochberg(z, fdr, mu0=0., sigma0=1.):
    '''Performs Benjamini-Hochberg multiple hypothesis testing on z at the given false discovery rate threshold.'''
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
    if z_shape is not None:
        x = np.zeros(z.shape)
        x[discoveries] = 1
        discoveries = np.where(x.reshape(z_shape) == 1)
    return discoveries


# In[3]:


# normix.py
import numpy as np
from scipy.stats import norm as norm
from scipy.optimize import fmin_bfgs
from copy import deepcopy

class GridDistribution:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def pdf(self, data):
        # Find the closest bins
        rhs = np.searchsorted(self.x, data)
        lhs = (rhs - 1).clip(0)
        rhs = rhs.clip(0, len(self.x) - 1)

        # Linear approximation (trapezoid rule)
        rhs_dist = np.abs(self.x[rhs] - data)
        lhs_dist = np.abs(self.x[lhs] - data)
        denom = rhs_dist + lhs_dist
        denom[denom == 0] = 1. # handle the zero-distance edge-case
        rhs_weight = 1.0 - rhs_dist / denom
        lhs_weight = 1.0 - rhs_weight

        return lhs_weight * self.y[lhs] + rhs_weight * self.y[rhs]

def trapezoid(x, y):
    return np.sum((x[1:] - x[0:-1]) * (y[1:] + y[0:-1]) / 2.)

def generate_sweeps(num_sweeps, num_samples):
    results = []
    for sweep in range(num_sweeps):
        a = np.arange(num_samples)
        np.random.shuffle(a)
        results.extend(a)
    return np.array(results)

def predictive_recursion(z, num_sweeps, grid_x, mu0=0., sig0=1.,
                            nullprob=1.0, decay=-0.67):
    sweeporder = generate_sweeps(num_sweeps, len(z))
    theta_guess = np.ones(len(grid_x)) / float(len(grid_x))
    return predictive_recursion_fdr(z, sweeporder, grid_x, theta_guess,
                                    mu0, sig0, nullprob, decay)

def predictive_recursion_fdr(z, sweeporder, grid_x, theta_guess, mu0 = 0.,
                            sig0 = 1.0, nullprob = 1.0, decay = -0.67):
    gridsize = grid_x.shape[0]
    theta_subdens = deepcopy(theta_guess)
    pi0 = nullprob
    joint1 = np.zeros(gridsize)
    ftheta1 = np.zeros(gridsize)

    # Begin sweep through the data
    for i, k in enumerate(sweeporder):
        cc = (3. + i)**decay
        joint1 = norm.pdf(grid_x, loc=z[k] - mu0, scale=sig0) * theta_subdens
        m0 = pi0 * norm.pdf(z[k] - mu0, 0., sig0)
        m1 = trapezoid(grid_x, joint1)
        mmix = m0 + m1
        pi0 = (1. - cc) * pi0 + cc * (m0 / mmix)
        ftheta1 = joint1 / mmix
        theta_subdens = (1. - cc) * theta_subdens + cc * ftheta1

    # Now calculate marginal distribution along the grid points
    y_mix = np.zeros(gridsize)
    y_signal = np.zeros(gridsize)
    for i, x in enumerate(grid_x):
        joint1 = norm.pdf(grid_x, x - mu0, sig0) * theta_subdens
        m0 = pi0 * norm.pdf(x, mu0, sig0)
        m1 = trapezoid(grid_x, joint1)
        y_mix[i] = m0 + m1;
        y_signal[i] = m1 / (1. - pi0)

    return {'grid_x': grid_x,
            'sweeporder': sweeporder,
            'theta_subdens': theta_subdens,
            'pi0': pi0,
            'y_mix': y_mix,
            'y_signal': y_signal}

def empirical_null(z, nmids=150, pct=-0.01, pct0=0.25, df=4, verbose=0):
    '''Estimate f(z) and f_0(z) using a polynomial approximation to Efron (2004)'s method.'''
    N = len(z)
    med = np.median(z)
    lb = med + (1 - pct) * (z.min() - med)
    ub = med + (1 - pct) * (z.max() - med)

    breaks = np.linspace(lb, ub, nmids+1)
    zcounts = np.histogram(z, bins=breaks)[0]
    mids = (breaks[:-1] + breaks[1:])/2

    ### Truncated Polynomial

    # Truncate to [-3, 3]
    selected = np.logical_and(mids >= -3, mids <= 3)
    zcounts = zcounts[selected]
    mids = mids[selected]

    # Form a polynomial basis and multiply by z-counts
    X = np.array([mids ** i for i in range(df+1)]).T
    beta0 = np.zeros(df+1)
    loglambda_loss = lambda beta, X, y: -((X * y[:,np.newaxis]).dot(beta) - np.exp(X.dot(beta).clip(-20,20))).sum() + 1e-6*np.sqrt((beta ** 2).sum())
    results = fmin_bfgs(loglambda_loss, beta0, args=(X, zcounts), disp=verbose)
    a = np.linspace(-3,3,1000)
    B = np.array([a ** i for i in range(df+1)]).T
    beta_hat = results

    # Back out the mean and variance from the Taylor terms
    x_max = mids[np.argmax(X.dot(beta_hat))]
    loglambda_deriv1_atmode = np.array([i * beta_hat[i] * x_max**(i-1) for i in range(1,df+1)]).sum()
    loglambda_deriv2_atmode = np.array([i * (i-1) * beta_hat[i] * x_max**(i-2) for i in range(2,df+1)]).sum()
    
    # Handle the edge cases that arise with numerical precision issues
    sigma_enull = np.sqrt(-1.0/loglambda_deriv2_atmode) if loglambda_deriv2_atmode < 0 else 1
    mu_enull = (x_max - loglambda_deriv1_atmode/loglambda_deriv2_atmode) if loglambda_deriv2_atmode != 0 else 0

    return (mu_enull, sigma_enull)


# In[4]:


#smoothed_fdr.py
import itertools
import numpy as np
from scipy import sparse
from scipy.stats import norm
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse import csc_matrix, linalg as sla
from functools import partial
from collections import deque
from pygfl.solver import TrailSolver

class GaussianKnown:
    '''
    A simple Gaussian distribution with known mean and stdev.
    '''
    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev

    def pdf(self, data):
        return norm.pdf(data, loc=self.mean, scale=self.stdev)

    def sample(self):
        return np.random.normal(loc=self.mean, scale=self.stdev)

    def noisy_pdf(self, data):
        return norm.pdf(data, loc=self.mean, scale=np.sqrt(self.stdev**2 + 1))

    def __repr__(self):
        return 'N({0}, {1}^2)'.format(self.mean, self.stdev)


class SmoothedFdr(object):
    def __init__(self, signal_dist, null_dist, penalties_cross_x=None):
        self.signal_dist = signal_dist
        self.null_dist = null_dist

        if penalties_cross_x is None:
            self.penalties_cross_x = np.dot
        else:
            self.penalties_cross_x = penalties_cross_x

        self.w_iters = []
        self.beta_iters = []
        self.c_iters = []
        self.delta_iters = []

        # ''' Load the graph fused lasso library '''
        # graphfl_lib = cdll.LoadLibrary('libgraphfl.so')
        # self.graphfl_weight = graphfl_lib.graph_fused_lasso_weight_warm
        # self.graphfl_weight.restype = c_int
        # self.graphfl_weight.argtypes = [c_int, ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
        #                     c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'),
        #                     c_double, c_double, c_double, c_int, c_double,
        #                     ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]
        self.solver = TrailSolver()

    def add_step(self, w, beta, c, delta):
        self.w_iters.append(w)
        self.beta_iters.append(beta)
        self.c_iters.append(c)
        self.delta_iters.append(delta)

    def finish(self):
        self.w_iters = np.array(self.w_iters)
        self.beta_iters = np.array(self.beta_iters)
        self.c_iters = np.array(self.c_iters)
        self.delta_iters = np.array(self.delta_iters)

    def reset(self):
        self.w_iters = []
        self.beta_iters = []
        self.c_iters = []
        self.delta_iters = []

    def solution_path(self, data, penalties, dof_tolerance=1e-4,
            min_lambda=0.20, max_lambda=1.5, lambda_bins=30,
            converge=0.00001, max_steps=100, m_converge=0.00001,
            m_max_steps=20, cd_converge=0.00001, cd_max_steps=1000, verbose=0, dual_solver='graph',
            admm_alpha=1., admm_inflate=2., admm_adaptive=False, initial_values=None,
            grid_data=None, grid_map=None):
        '''Follows the solution path of the generalized lasso to find the best lambda value.'''
        lambda_grid = np.exp(np.linspace(np.log(max_lambda), np.log(min_lambda), lambda_bins))
        aic_trace = np.zeros(lambda_grid.shape) # The AIC score for each lambda value
        aicc_trace = np.zeros(lambda_grid.shape) # The AICc score for each lambda value (correcting for finite sample size)
        bic_trace = np.zeros(lambda_grid.shape) # The BIC score for each lambda value
        dof_trace = np.zeros(lambda_grid.shape) # The degrees of freedom of each final solution
        log_likelihood_trace = np.zeros(lambda_grid.shape)
        beta_trace = []
        u_trace = []
        w_trace = []
        c_trace = []
        results_trace = []
        best_idx = None
        best_plateaus = None
        flat_data = data.flatten()
        edges = penalties[3] if dual_solver == 'graph' else None
        if grid_data is not None:
                grid_points = np.zeros(grid_data.shape)
                grid_points[:,:] = np.nan
        for i, _lambda in enumerate(lambda_grid):
            if verbose:
                print('#{0} Lambda = {1}'.format(i, _lambda))

            # Clear out all the info from the previous run
            self.reset()

            # Fit to the final values
            results = self.run(flat_data, penalties, _lambda=_lambda, converge=converge, max_steps=max_steps,
                           m_converge=m_converge, m_max_steps=m_max_steps, cd_converge=cd_converge,
                           cd_max_steps=cd_max_steps, verbose=verbose, dual_solver=dual_solver,
                           admm_alpha=admm_alpha, admm_inflate=admm_inflate, admm_adaptive=admm_adaptive,
                           initial_values=initial_values)

            if verbose:
                print('Calculating degrees of freedom')

            # Create a grid structure out of the vector of betas
            if grid_map is not None:
                grid_points[grid_map != -1] = results['beta'][grid_map[grid_map != -1]]
            else:
                grid_points = results['beta'].reshape(data.shape)

            # Count the number of free parameters in the grid (dof)
            plateaus = calc_plateaus(grid_points, dof_tolerance, edges=edges)
            dof_trace[i] = len(plateaus)
            #dof_trace[i] = (np.abs(penalties.dot(results['beta'])) >= dof_tolerance).sum() + 1 # Use the naive DoF

            if verbose:
                print('Calculating AIC')

            # Get the negative log-likelihood
            log_likelihood_trace[i] = -self._data_negative_log_likelihood(flat_data, results['c'])

            # Calculate AIC = 2k - 2ln(L)
            aic_trace[i] = 2. * dof_trace[i] - 2. * log_likelihood_trace[i]
            
            # Calculate AICc = AIC + 2k * (k+1) / (n - k - 1)
            aicc_trace[i] = aic_trace[i] + 2 * dof_trace[i] * (dof_trace[i]+1) / (flat_data.shape[0] - dof_trace[i] - 1.)

            # Calculate BIC = -2ln(L) + k * (ln(n) - ln(2pi))
            bic_trace[i] = -2 * log_likelihood_trace[i] + dof_trace[i] * (np.log(len(flat_data)) - np.log(2 * np.pi))

            # Track the best model thus far
            if best_idx is None or bic_trace[i] < bic_trace[best_idx]:
                best_idx = i
                best_plateaus = plateaus

            # Save the final run parameters to use for warm-starting the next iteration
            initial_values = results

            # Save the trace of all the resulting parameters
            beta_trace.append(results['beta'])
            u_trace.append(results['u'])
            w_trace.append(results['w'])
            c_trace.append(results['c'])

            if verbose:
                print('DoF: {0} AIC: {1} AICc: {2} BIC: {3}'.format(dof_trace[i], aic_trace[i], aicc_trace[i], bic_trace[i]))

        if verbose:
            print('Best setting (by BIC): lambda={0} [DoF: {1}, AIC: {2}, AICc: {3} BIC: {4}]'.format(lambda_grid[best_idx], dof_trace[best_idx], aic_trace[best_idx], aicc_trace[best_idx], bic_trace[best_idx]))

        return {'aic': aic_trace,
                'aicc': aicc_trace,
                'bic': bic_trace,
                'dof': dof_trace,
                'loglikelihood': log_likelihood_trace,
                'beta': np.array(beta_trace),
                'u': np.array(u_trace),
                'w': np.array(w_trace),
                'c': np.array(c_trace),
                'lambda': lambda_grid,
                'best': best_idx,
                'plateaus': best_plateaus}


    def run(self, data, penalties, _lambda=0.1, converge=0.00001, max_steps=100, m_converge=0.00001,
            m_max_steps=100, cd_converge=0.00001, cd_max_steps=100, verbose=0, dual_solver='graph',
            admm_alpha=1., admm_inflate=2., admm_adaptive=False, initial_values=None):
        '''Runs the Expectation-Maximization algorithm for the data with the given penalty matrix.'''
        delta = converge + 1
        
        if initial_values is None:
            beta = np.zeros(data.shape)
            prior_prob = np.exp(beta) / (1 + np.exp(beta))
            u = initial_values
        else:
            beta = initial_values['beta']
            prior_prob = initial_values['c']
            u = initial_values['u']

        prev_nll = 0
        cur_step = 0
        
        while delta > converge and cur_step < max_steps:
            if verbose:
                print('Step #{0}'.format(cur_step))

            if verbose:
                print('\tE-step...')

            # Get the likelihood weights vector (E-step)
            post_prob = self._e_step(data, prior_prob)

            if verbose:
                print('\tM-step...')

            # Find beta using an alternating Taylor approximation and convex optimization (M-step)
            beta, u = self._m_step(beta, prior_prob, post_prob, penalties, _lambda,
                                   m_converge, m_max_steps,
                                   cd_converge, cd_max_steps,
                                   verbose, dual_solver,
                                   admm_adaptive=admm_adaptive,
                                   admm_inflate=admm_inflate,
                                   admm_alpha=admm_alpha,
                                   u0=u)

            # Get the signal probabilities
            prior_prob = ilogit(beta)
            cur_nll = self._data_negative_log_likelihood(data, prior_prob)

            if dual_solver == 'admm':
                # Get the negative log-likelihood of the data given our new parameters
                cur_nll += _lambda * np.abs(u['r']).sum()
            
            # Track the change in log-likelihood to see if we've converged
            delta = np.abs(cur_nll - prev_nll) / (prev_nll + converge)

            if verbose:
                print('\tDelta: {0}'.format(delta))

            # Track the step
            self.add_step(post_prob, beta, prior_prob, delta)

            # Increment the step counter
            cur_step += 1

            # Update the negative log-likelihood tracker
            prev_nll = cur_nll

            # DEBUGGING
            if verbose:
                print('\tbeta: [{0:.4f}, {1:.4f}]'.format(beta.min(), beta.max()))
                print('\tprior_prob:    [{0:.4f}, {1:.4f}]'.format(prior_prob.min(), prior_prob.max()))
                print('\tpost_prob:    [{0:.4f}, {1:.4f}]'.format(post_prob.min(), post_prob.max()))
                if dual_solver != 'graph':
                    print('\tdegrees of freedom: {0}'.format((np.abs(penalties.dot(beta)) >= 1e-4).sum()))

        # Return the results of the run
        return {'beta': beta, 'u': u, 'w': post_prob, 'c': prior_prob}

    def _data_negative_log_likelihood(self, data, prior_prob):
        '''Calculate the negative log-likelihood of the data given the weights.'''
        signal_weight = prior_prob * self.signal_dist.pdf(data)
        null_weight = (1-prior_prob) * self.null_dist.pdf(data)
        return -np.log(signal_weight + null_weight).sum()

    def _e_step(self, data, prior_prob):
        '''Calculate the complete-data sufficient statistics (weights vector).'''
        signal_weight = prior_prob * self.signal_dist.pdf(data)
        null_weight = (1-prior_prob) * self.null_dist.pdf(data)
        post_prob = signal_weight / (signal_weight + null_weight)
        return post_prob

    def _m_step(self, beta, prior_prob, post_prob, penalties,
                _lambda, converge, max_steps,
                cd_converge, cd_max_steps,
                verbose, dual_solver, u0=None,
                admm_alpha=1., admm_inflate=2., admm_adaptive=False):
        '''
        Alternating Second-order Taylor-series expansion about the current iterate
        and coordinate descent to optimize Beta.
        '''
        prev_nll = self._m_log_likelihood(post_prob, beta)
        delta = converge + 1
        u = u0
        cur_step = 0
        while delta > converge and cur_step < max_steps:
            if verbose > 1:
                print('\t\tM-Step iteration #{0}'.format(cur_step))
                print('\t\tTaylor approximation...')

            # Cache the exponentiated beta
            exp_beta = np.exp(beta)

            # Form the parameters for our weighted least squares
            if dual_solver != 'admm' and dual_solver != 'graph':
                # weights is a diagonal matrix, represented as a vector for efficiency
                weights = 0.5 * exp_beta / (1 + exp_beta)**2
                y = (1+exp_beta)**2 * post_prob / exp_beta + beta - (1 + exp_beta)
                if verbose > 1:
                    print('\t\tForming dual...')
                x = np.sqrt(weights) * y
                A = (1. / np.sqrt(weights))[:,np.newaxis] * penalties.T
            else:
                weights = (prior_prob * (1 - prior_prob))
                y = beta - (prior_prob - post_prob) / weights
                print(weights)
                print(y)

            if dual_solver == 'cd':
                # Solve the dual via coordinate descent
                u = self._u_coord_descent(x, A, _lambda, cd_converge, cd_max_steps, verbose > 1, u0=u)
            elif dual_solver == 'sls':
                # Solve the dual via sequential least squares
                u = self._u_slsqp(x, A, _lambda, verbose > 1, u0=u)
            elif dual_solver == 'lbfgs':
                # Solve the dual via L-BFGS-B
                u = self._u_lbfgsb(x, A, _lambda, verbose > 1, u0=u)
            elif dual_solver == 'admm':
                # Solve the dual via alternating direction methods of multipliers
                #u = self._u_admm_1dfusedlasso(y, weights, _lambda, cd_converge, cd_max_steps, verbose > 1, initial_values=u)
                #u = self._u_admm(y, weights, _lambda, penalties, cd_converge, cd_max_steps, verbose > 1, initial_values=u)
                u = self._u_admm_lucache(y, weights, _lambda, penalties, cd_converge, cd_max_steps,
                                        verbose > 1, initial_values=u, inflate=admm_inflate,
                                        adaptive=admm_adaptive, alpha=admm_alpha)
                beta = u['x']
            elif dual_solver == 'graph':
                u = self._graph_fused_lasso(y, weights, _lambda, penalties[0], penalties[1], penalties[2], penalties[3], cd_converge, cd_max_steps, max(0, verbose - 1), admm_alpha, admm_inflate, initial_values=u)
                beta = u['beta']
                # if np.abs(beta).max() > 20:
                #     beta = np.clip(beta, -20, 20)
                #     u = None
            else:
                raise Exception('Unknown solver: {0}'.format(dual_solver))

            if dual_solver != 'admm' and dual_solver != 'graph':
                # Back out beta from the dual solution
                beta = y - (1. / weights) * penalties.T.dot(u)

            # Get the current log-likelihood
            cur_nll = self._m_log_likelihood(post_prob, beta)

            # Track the convergence
            delta = np.abs(prev_nll - cur_nll) / (prev_nll + converge)

            if verbose > 1:
                print('\t\tM-step delta: {0}'.format(delta))

            # Increment the step counter
            cur_step += 1

            # Update the negative log-likelihood tracker
            prev_nll = cur_nll

        return beta, u
    
    def _m_log_likelihood(self, post_prob, beta):
        '''Calculate the log-likelihood of the betas given the weights and data.'''
        return (np.log(1 + np.exp(beta)) - post_prob * beta).sum()

    def _graph_fused_lasso(self, y, weights, _lambda, ntrails, trails, breakpoints, edges, converge, max_steps, verbose, alpha, inflate, initial_values=None):
        '''Solve for u using a super fast graph fused lasso library that has an optimized ADMM routine.'''
        if verbose:
            print('\t\tSolving via Graph Fused Lasso')
        # if initial_values is None:
        #     beta = np.zeros(y.shape, dtype='double')
        #     z = np.zeros(breakpoints[-1], dtype='double')
        #     u = np.zeros(breakpoints[-1], dtype='double')
        # else:
        #     beta = initial_values['beta']
        #     z = initial_values['z']
        #     u = initial_values['u']
        # n = y.shape[0]
        # self.graphfl_weight(n, y, weights, ntrails, trails, breakpoints, _lambda, alpha, inflate, max_steps, converge, beta, z, u)
        # return {'beta': beta, 'z': z, 'u': u }
        self.solver.alpha = alpha
        self.solver.inflate = inflate
        self.solver.maxsteps = max_steps
        self.solver.converge = converge
        self.solver.set_data(y, edges, ntrails, trails, breakpoints, weights=weights)
        if initial_values is not None:
            self.solver.beta = initial_values['beta']
            self.solver.z = initial_values['z']
            self.solver.u = initial_values['u']
        self.solver.solve(_lambda)
        return {'beta': self.solver.beta, 'z': self.solver.z, 'u': self.solver.u }
        

    def _u_admm_lucache(self, y, weights, _lambda, D, converge_threshold, max_steps, verbose, alpha=1.8, initial_values=None, inflate=2., adaptive=False):
        '''Solve for u using alternating direction method of multipliers with a cached LU decomposition.'''
        if verbose:
            print('\t\tSolving u via Alternating Direction Method of Multipliers')

        n = len(y)
        m = D.shape[0]
        a = inflate * _lambda # step-size parameter

        # Initialize primal and dual variables from warm start
        if initial_values is None:
            # Graph Laplacian
            L = csc_matrix(D.T.dot(D) + csc_matrix(np.eye(n)))

            # Cache the LU decomposition
            lu_factor = sla.splu(L, permc_spec='MMD_AT_PLUS_A')
            
            x = np.array([y.mean()] * n) # likelihood term
            z = np.zeros(n) # slack variable for likelihood
            r = np.zeros(m) # penalty term
            s = np.zeros(m) # slack variable for penalty
            u_dual = np.zeros(n) # scaled dual variable for constraint x = z
            t_dual = np.zeros(m) # scaled dual variable for constraint r = s
        else:
            lu_factor = initial_values['lu_factor']
            x = initial_values['x']
            z = initial_values['z']
            r = initial_values['r']
            s = initial_values['s']
            u_dual = initial_values['u_dual']
            t_dual = initial_values['t_dual']

        primal_trace = []
        dual_trace = []
        converged = False
        cur_step = 0
        D_full = D
        while not converged and cur_step < max_steps:
            # Update x
            x = (weights * y + a * (z - u_dual)) / (weights + a)
            x_accel = alpha * x + (1 - alpha) * z # over-relaxation

            # Update constraint term r
            arg = s - t_dual
            local_lambda = (_lambda - np.abs(arg) / 2.).clip(0) if adaptive else _lambda
            r = _soft_threshold(arg, local_lambda / a)
            r_accel = alpha * r + (1 - alpha) * s

            # Projection to constraint set
            arg = x_accel + u_dual + D.T.dot(r_accel + t_dual)
            z_new = lu_factor.solve(arg)
            s_new = D.dot(z_new)
            dual_residual_u = a * (z_new - z)
            dual_residual_t = a * (s_new - s)
            z = z_new
            s = s_new

            # Dual update
            primal_residual_x = x_accel - z
            primal_residual_r = r_accel - s
            u_dual = u_dual + primal_residual_x
            t_dual = t_dual + primal_residual_r

            # Check convergence
            primal_resnorm = np.sqrt((np.array([i for i in primal_residual_x] + [i for i in primal_residual_r])**2).mean())
            dual_resnorm = np.sqrt((np.array([i for i in dual_residual_u] + [i for i in dual_residual_t])**2).mean())
            primal_trace.append(primal_resnorm)
            dual_trace.append(dual_resnorm)
            converged = dual_resnorm < converge_threshold and primal_resnorm < converge_threshold

            if primal_resnorm > 5 * dual_resnorm:
                a *= inflate
                u_dual /= inflate
                t_dual /= inflate
            elif dual_resnorm > 5 * primal_resnorm:
                a /= inflate
                u_dual *= inflate
                t_dual *= inflate

            # Update the step counter
            cur_step += 1

            if verbose and cur_step % 100 == 0:
                print('\t\t\tStep #{0}: dual_resnorm: {1:.6f} primal_resnorm: {2:.6f}'.format(cur_step, dual_resnorm, primal_resnorm))

        return {'x': x, 'r': r, 'z': z, 's': s, 'u_dual': u_dual, 't_dual': t_dual,
                'primal_trace': primal_trace, 'dual_trace': dual_trace, 'steps': cur_step,
                'lu_factor': lu_factor}

    def _u_admm(self, y, weights, _lambda, D, converge_threshold, max_steps, verbose, alpha=1.0, initial_values=None):
        '''Solve for u using alternating direction method of multipliers.'''
        if verbose:
            print('\t\tSolving u via Alternating Direction Method of Multipliers')

        n = len(y)
        m = D.shape[0]

        a = _lambda # step-size parameter

        # Set up system involving graph Laplacian
        L = D.T.dot(D)
        W_over_a = np.diag(weights / a)
        x_denominator = W_over_a + L
        #x_denominator = sparse.linalg.inv(W_over_a + L)

        # Initialize primal and dual variables
        if initial_values is None:
            x = np.array([y.mean()] * n)
            z = np.zeros(m)
            u = np.zeros(m)
        else:
            x = initial_values['x']
            z = initial_values['z']
            u = initial_values['u']

        primal_trace = []
        dual_trace = []
        converged = False
        cur_step = 0
        while not converged and cur_step < max_steps:
            # Update x
            x_numerator = 1.0 / a * weights * y + D.T.dot(a * z - u)
            x = np.linalg.solve(x_denominator, x_numerator)
            Dx = D.dot(x)

            # Update z
            Dx_relaxed = alpha * Dx + (1 - alpha) * z # over-relax Dx
            z_new = _soft_threshold(Dx_relaxed + u / a, _lambda / a)
            dual_residual = a * D.T.dot(z_new - z)
            z = z_new
            primal_residual = Dx_relaxed - z

            # Update u
            u = u + a * primal_residual

            # Check convergence
            primal_resnorm = np.sqrt((primal_residual ** 2).mean())
            dual_resnorm = np.sqrt((dual_residual ** 2).mean())
            primal_trace.append(primal_resnorm)
            dual_trace.append(dual_resnorm)
            converged = dual_resnorm < converge_threshold and primal_resnorm < converge_threshold

            # Update step-size parameter based on norm of primal and dual residuals
            # This is the varying penalty extension to standard ADMM
            a *= 2 if primal_resnorm > 10 * dual_resnorm else 0.5

            # Recalculate the x_denominator since we changed the step-size
            # TODO: is this worth it? We're paying a matrix inverse in exchange for varying the step size
            #W_over_a = sparse.dia_matrix(np.diag(weights / a))
            W_over_a = np.diag(weights / a)
            #x_denominator = sparse.linalg.inv(W_over_a + L)
            
            # Update the step counter
            cur_step += 1

            if verbose and cur_step % 100 == 0:
                print('\t\t\tStep #{0}: dual_resnorm: {1:.6f} primal_resnorm: {2:.6f}'.format(cur_step, dual_resnorm, primal_resnorm))

        dof = np.sum(Dx > converge_threshold) + 1.
        AIC = np.sum((y - x)**2) + 2 * dof

        return {'x': x, 'z': z, 'u': u, 'dof': dof, 'AIC': AIC}

    def _u_admm_1dfusedlasso(self, y, W, _lambda, converge_threshold, max_steps, verbose, alpha=1.0, initial_values=None):
        '''Solve for u using alternating direction method of multipliers. Note that this method only works for the 1-D fused lasso case.'''
        if verbose:
            print('\t\tSolving u via Alternating Direction Method of Multipliers (1-D fused lasso)')

        n = len(y)
        m = n - 1

        a = _lambda

        # The D matrix is the first-difference operator. K is the matrix (W + a D^T D)
        # where W is the diagonal matrix of weights. We use a tridiagonal representation
        # of K.
        Kd = np.array([a] + [2*a] * (n-2) + [a]) + W # diagonal entries
        Kl = np.array([-a] * (n-1)) # below the diagonal
        Ku = np.array([-a] * (n-1)) # above the diagonal

        # Initialize primal and dual variables
        if initial_values is None:
            x = np.array([y.mean()] * n)
            z = np.zeros(m)
            u = np.zeros(m)
        else:
            x = initial_values['x']
            z = initial_values['z']
            u = initial_values['u']

        primal_trace = []
        dual_trace = []
        converged = False
        cur_step = 0
        while not converged and cur_step < max_steps:
            # Update x
            out = _1d_fused_lasso_crossprod(a*z - u)
            x = tridiagonal_solve(Kl, Ku, Kd, W * y + out)
            Dx = np.ediff1d(x)

            # Update z
            Dx_hat = alpha * Dx + (1 - alpha) * z # Over-relaxation
            z_new = _soft_threshold(Dx_hat + u / a, _lambda / a)
            dual_residual = a * _1d_fused_lasso_crossprod(z_new - z)
            z = z_new
            primal_residual = Dx - z
            #primal_residual = Dx_hat - z

            # Update u
            u = (u + a * primal_residual).clip(-_lambda, _lambda)

            # Check convergence
            primal_resnorm = np.sqrt((primal_residual ** 2).mean())
            dual_resnorm = np.sqrt((dual_residual ** 2).mean())
            primal_trace.append(primal_resnorm)
            dual_trace.append(dual_resnorm)
            converged = dual_resnorm < converge_threshold and primal_resnorm < converge_threshold
            
            # Update step-size parameter based on norm of primal and dual residuals
            a *= 2 if primal_resnorm > 10 * dual_resnorm else 0.5
            Kd = np.array([a] + [2*a] * (n-2) + [a]) + W # diagonal entries
            Kl = np.array([-a] * (n-1)) # below the diagonal
            Ku = np.array([-a] * (n-1)) # above the diagonal

            cur_step += 1

            if verbose and cur_step % 100 == 0:
                print('\t\t\tStep #{0}: dual_resnorm: {1:.6f} primal_resnorm: {2:.6f}'.format(cur_step, dual_resnorm, primal_resnorm))

        dof = np.sum(Dx > converge_threshold) + 1.
        AIC = np.sum((y - x)**2) + 2 * dof

        return {'x': x, 'z': z, 'u': u, 'dof': dof, 'AIC': AIC}


    def _u_coord_descent(self, x, A, _lambda, converge, max_steps, verbose, u0=None):
        '''Solve for u using coordinate descent.'''
        if verbose:
            print('\t\tSolving u via Coordinate Descent')
        
        u = u0 if u0 is not None else np.zeros(A.shape[1])

        l2_norm_A = (A * A).sum(axis=0)
        r = x - A.dot(u)
        delta = converge + 1
        prev_objective = _u_objective_func(u, x, A)
        cur_step = 0
        while delta > converge and cur_step < max_steps:
            # Update each coordinate one at a time.
            for coord in range(len(u)):
                prev_u = u[coord]
                next_u = prev_u + A.T[coord].dot(r) / l2_norm_A[coord]
                u[coord] = min(_lambda, max(-_lambda, next_u))
                r += A.T[coord] * prev_u - A.T[coord] * u[coord]

            # Track the change in the objective function value
            cur_objective = _u_objective_func(u, x, A)
            delta = np.abs(prev_objective - cur_objective) / (prev_objective + converge)

            if verbose and cur_step % 100 == 0:
                print('\t\t\tStep #{0}: Objective: {1:.6f} CD Delta: {2:.6f}'.format(cur_step, cur_objective, delta))

            # Increment the step counter and update the previous objective value
            cur_step += 1
            prev_objective = cur_objective

        return u

    def _u_slsqp(self, x, A, _lambda, verbose, u0=None):
        '''Solve for u using sequential least squares.'''
        if verbose:
            print('\t\tSolving u via Sequential Least Squares')

        if u0 is None:
            u0 = np.zeros(A.shape[1])

        # Create our box constraints
        bounds = [(-_lambda, _lambda) for u0_i in u0]

        results = minimize(_u_objective_func, u0,
                           args=(x, A),
                           jac=_u_objective_deriv,
                           bounds=bounds,
                           method='SLSQP',
                           options={'disp': False, 'maxiter': 1000})

        if verbose:
            print('\t\t\t{0}'.format(results.message))
            print('\t\t\tFunction evaluations: {0}'.format(results.nfev))
            print('\t\t\tGradient evaluations: {0}'.format(results.njev))
            print('\t\t\tu: [{0}, {1}]'.format(results.x.min(), results.x.max()))

        return results.x

    def _u_lbfgsb(self, x, A, _lambda, verbose, u0=None):
        '''Solve for u using L-BFGS-B.'''
        if verbose:
            print('\t\tSolving u via L-BFGS-B')

        if u0 is None:
            u0 = np.zeros(A.shape[1])

        # Create our box constraints
        bounds = [(-_lambda, _lambda) for _ in u0]

        # Fit
        results = minimize(_u_objective_func, u0, args=(x, A), method='L-BFGS-B', bounds=bounds, options={'disp': verbose})

        return results.x

    def plateau_regression(self, plateaus, data, grid_map=None, verbose=False):
        '''Perform unpenalized 1-d regression for each of the plateaus.'''
        weights = np.zeros(data.shape)
        for i,(level,p) in enumerate(plateaus):
            if verbose:
                print('\tPlateau #{0}'.format(i+1))
            
            # Get the subset of grid points for this plateau
            if grid_map is not None:
                plateau_data = np.array([data[grid_map[x,y]] for x,y in p])
            else:
                plateau_data = np.array([data[x,y] for x,y in p])

            w = single_plateau_regression(plateau_data, self.signal_dist, self.null_dist)
            for idx in p:
                weights[idx if grid_map is None else grid_map[idx[0], idx[1]]] = w
        posteriors = self._e_step(data, weights)
        weights = weights.flatten()
        return (weights, posteriors)


def _u_objective_func(u, x, A):
    return np.linalg.norm(x - A.dot(u))**2

def _u_objective_deriv(u, x, A):
    return 2*A.T.dot(A.dot(u) - x)

def _u_slsqp_constraint_func(idx, _lambda, u):
    '''Constraint function for the i'th value of u.'''
    return np.array([_lambda - np.abs(u[idx])])

def _u_slsqp_constraint_deriv(idx, u):
    jac = np.zeros(len(u))
    jac[idx] = -np.sign(u[idx])
    return jac

def _1d_fused_lasso_crossprod(x):
    '''Efficiently compute the cross-product D^T x, where D is the first-differences matrix.'''
    return -np.ediff1d(x, to_begin=x[0], to_end=-x[-1])

def _soft_threshold(x, _lambda):
    return np.sign(x) * (np.abs(x) - _lambda).clip(0)

## Tri-Diagonal Matrix Algorithm (a.k.a Thomas algorithm) solver
## Source: http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
def tridiagonal_solve(a,b,c,f):
    alpha = [0]
    beta = [0]
    n = len(f)
    x = [0] * n
 
    for i in range(n-1):
        alpha.append(-b[i]/(a[i]*alpha[i] + c[i]))
        beta.append((f[i] - a[i]*beta[i])/(a[i]*alpha[i] + c[i]))
 
    x[n-1] = (f[n-1] - a[n-2]*beta[n-1])/(c[n-1] + a[n-2]*alpha[n-1])
 
    for i in reversed(range(n-1)):
        x[i] = alpha[i+1]*x[i+1] + beta[i+1]
 
    return np.array(x)

def ilogit(x):
    return 1. / (1. + np.exp(-x))

def calc_plateaus(beta, rel_tol=1e-4, edges=None, verbose=0):
    '''Calculate the plateaus (degrees of freedom) of a 1d or 2d grid of beta values in linear time.'''
    to_check = deque(itertools.product(*[range(x) for x in beta.shape])) if edges is None else deque(range(len(beta)))
    check_map = np.zeros(beta.shape, dtype=bool)
    check_map[np.isnan(beta)] = True
    plateaus = []

    if verbose:
        print('\tCalculating plateaus...')

    if verbose > 1:
        print('\tIndices to check {0} {1}'.format(len(to_check), check_map.shape))

    # Loop until every beta index has been checked
    while to_check:
        if verbose > 1:
            print('\t\tPlateau #{0}'.format(len(plateaus) + 1))

        # Get the next unchecked point on the grid
        idx = to_check.popleft()

        # If we already have checked this one, just pop it off
        while to_check and check_map[idx]:
            try:
                idx = to_check.popleft()
            except:
                break

        # Edge case -- If we went through all the indices without reaching an unchecked one.
        if check_map[idx]:
            break

        # Create the plateau and calculate the inclusion conditions
        cur_plateau = set([idx])
        cur_unchecked = deque([idx])
        val = beta[idx]
        min_member = val - rel_tol
        max_member = val + rel_tol

        # Check every possible boundary of the plateau
        while cur_unchecked:
            idx = cur_unchecked.popleft()
            
            # neighbors to check
            local_check = []

            # Generic graph case
            if edges is not None:
                local_check.extend(edges[idx])

            # 1d case -- check left and right
            elif len(beta.shape) == 1:
                if idx[0] > 0:
                    local_check.append(idx[0] - 1) # left
                if idx[0] < beta.shape[0] - 1:
                    local_check.append(idx[0] + 1) # right

            # 2d case -- check left, right, up, and down
            elif len(beta.shape) == 2:
                if idx[0] > 0:
                    local_check.append((idx[0] - 1, idx[1])) # left
                if idx[0] < beta.shape[0] - 1:
                    local_check.append((idx[0] + 1, idx[1])) # right
                if idx[1] > 0:
                    local_check.append((idx[0], idx[1] - 1)) # down
                if idx[1] < beta.shape[1] - 1:
                    local_check.append((idx[0], idx[1] + 1)) # up

            # Only supports 1d and 2d cases for now
            else:
                raise Exception('Degrees of freedom calculation does not currently support more than 2 dimensions unless edges are specified explicitly. ({0} given)'.format(len(beta.shape)))

            # Check the index's unchecked neighbors
            for local_idx in local_check:
                if not check_map[local_idx]                     and beta[local_idx] >= min_member                     and beta[local_idx] <= max_member:
                        # Label this index as being checked so it's not re-checked unnecessarily
                        check_map[local_idx] = True

                        # Add it to the plateau and the list of local unchecked locations
                        cur_unchecked.append(local_idx)
                        cur_plateau.add(local_idx)

        # Track each plateau's indices
        plateaus.append((val, cur_plateau))

    # Returns the list of plateaus and their values
    return plateaus

def plateau_loss_func(c, data, signal_dist, null_dist):
    '''The negative log-likelihood function for a plateau.'''
    return -np.log(c * signal_dist.pdf(data) + (1. - c) * null_dist.pdf(data)).sum()

def single_plateau_regression(data, signal_dist, null_dist):
    '''Perform unpenalized 1-d regression on all of the points in a plateau.'''
    return minimize_scalar(plateau_loss_func, args=(data, signal_dist, null_dist), bounds=(0,1), method='Bounded').x


# In[5]:


# easy.py

# import itertools
# from functools import partial
# from scipy.stats import norm
# from scipy.sparse import csc_matrix, linalg as sla
# from scipy import sparse
# from scipy.optimize import minimize, minimize_scalar
# from collections import deque, namedtuple
import numpy as np
from networkx import Graph
from pygfl.solver import TrailSolver
from pygfl.trails import decompose_graph, save_chains
from pygfl.utils import chains_to_trails, calc_plateaus, hypercube_edges

def smooth_fdr(data, fdr_level, edges=None, initial_values=None, verbose=0, null_dist=None, signal_dist=None, num_sweeps=10, missing_val=None):
    flat_data = data.flatten()
    nonmissing_flat_data = flat_data

    if edges is None:
        if verbose:
            print('Using default edge set of a grid of same shape as the data: {0}'.format(data.shape))
        edges = hypercube_edges(data.shape)
        if missing_val is not None:
            if verbose:
                print('Removing all data points whose data value is {0}'.format(missing_val))
            edges = [(e1,e2) for (e1,e2) in edges if flat_data[e1] != missing_val and flat_data[e2] != missing_val]
            nonmissing_flat_data = flat_data[flat_data != missing_val]

    # Decompose the graph into trails
    g = Graph()
    g.add_edges_from(edges)
    chains = decompose_graph(g, heuristic='greedy')
    ntrails, trails, breakpoints, edges = chains_to_trails(chains)

    if null_dist is None:
        # empirical null estimation
        mu0, sigma0 = empirical_null(nonmissing_flat_data, verbose=max(0,verbose-1))
    elif isinstance(null_dist,GaussianKnown):
        mu0, sigma0 = null_dist.mean, null_dist.stdev
    else:
        mu0, sigma0 = null_dist
    null_dist = GaussianKnown(mu0, sigma0)

    if verbose:
        print('Empirical null: {0}'.format(null_dist))

    # signal distribution estimation
    if verbose:
        print('Running predictive recursion for {0} sweeps'.format(num_sweeps))
    if signal_dist is None:
        grid_x = np.linspace(min(-20, nonmissing_flat_data.min() - 1), max(nonmissing_flat_data.max() + 1, 20), 220)
        pr_results = predictive_recursion(nonmissing_flat_data, num_sweeps, grid_x, mu0=mu0, sig0=sigma0)
        signal_dist = GridDistribution(pr_results['grid_x'], pr_results['y_signal'])

    if verbose:
        print('Smoothing priors via solution path algorithm')

    solver = TrailSolver()
    solver.set_data(flat_data, edges, ntrails, trails, breakpoints)

    results = solution_path_smooth_fdr(flat_data, solver, null_dist, signal_dist, verbose=max(0, verbose-1))

    results['discoveries'] = calc_fdr(results['posteriors'], fdr_level)
    results['null_dist'] = null_dist
    results['signal_dist'] = signal_dist

    # Reshape everything back to the original data shape
    results['betas'] = results['betas'].reshape(data.shape)
    results['priors'] = results['priors'].reshape(data.shape)
    results['posteriors'] = results['posteriors'].reshape(data.shape)
    results['discoveries'] = results['discoveries'].reshape(data.shape)
    results['beta_iters'] = np.array([x.reshape(data.shape) for x in results['beta_iters']])
    results['prior_iters'] = np.array([x.reshape(data.shape) for x in results['prior_iters']])
    results['posterior_iters'] = np.array([x.reshape(data.shape) for x in results['posterior_iters']])

    return results

def smooth_fdr_known_dists(data, fdr_level, null_dist, signal_dist, edges=None, initial_values=None, verbose=0, missing_val=None):
    '''FDR smoothing where the null and alternative distributions are known
    (and not necessarily Gaussian). Both must define the function pdf.'''
    flat_data = data.flatten()
    nonmissing_flat_data = flat_data

    if edges is None:
        if verbose:
            print('Using default edge set of a grid of same shape as the data: {0}'.format(data.shape))
        edges = hypercube_edges(data.shape)
        if missing_val is not None:
            if verbose:
                print('Removing all data points whose data value is {0}'.format(missing_val))
            edges = [(e1,e2) for (e1,e2) in edges if flat_data[e1] != missing_val and flat_data[e2] != missing_val]
            nonmissing_flat_data = flat_data[flat_data != missing_val]

    # Decompose the graph into trails
    g = Graph()
    g.add_edges_from(edges)
    chains = decompose_graph(g, heuristic='greedy')
    ntrails, trails, breakpoints, edges = chains_to_trails(chains)

    if verbose:
        print('Smoothing priors via solution path algorithm')

    solver = TrailSolver()
    solver.set_data(flat_data, edges, ntrails, trails, breakpoints)

    results = solution_path_smooth_fdr(flat_data, solver, null_dist, signal_dist, verbose=max(0, verbose-1))

    results['discoveries'] = calc_fdr(results['posteriors'], fdr_level)
    results['null_dist'] = null_dist
    results['signal_dist'] = signal_dist

    # Reshape everything back to the original data shape
    results['betas'] = results['betas'].reshape(data.shape)
    results['priors'] = results['priors'].reshape(data.shape)
    results['posteriors'] = results['posteriors'].reshape(data.shape)
    results['discoveries'] = results['discoveries'].reshape(data.shape)
    results['beta_iters'] = np.array([x.reshape(data.shape) for x in results['beta_iters']])
    results['prior_iters'] = np.array([x.reshape(data.shape) for x in results['prior_iters']])
    results['posterior_iters'] = np.array([x.reshape(data.shape) for x in results['posterior_iters']])

    return results

def solution_path_smooth_fdr(data, solver, null_dist, signal_dist, min_lambda=0.20, max_lambda=1.5, lambda_bins=30, verbose=0, initial_values=None):
        '''Follows the solution path of the generalized lasso to find the best lambda value.'''
        lambda_grid = np.exp(np.linspace(np.log(max_lambda), np.log(min_lambda), lambda_bins))
        aic_trace = np.zeros(lambda_grid.shape) # The AIC score for each lambda value
        aicc_trace = np.zeros(lambda_grid.shape) # The AICc score for each lambda value (correcting for finite sample size)
        bic_trace = np.zeros(lambda_grid.shape) # The BIC score for each lambda value
        dof_trace = np.zeros(lambda_grid.shape) # The degrees of freedom of each final solution
        log_likelihood_trace = np.zeros(lambda_grid.shape)
        beta_trace = []
        u_trace = []
        w_trace = []
        c_trace = []
        results_trace = []
        best_idx = None
        best_plateaus = None
        for i, _lambda in enumerate(lambda_grid):
            if verbose:
                print('#{0} Lambda = {1}'.format(i, _lambda))

            # Fit to the final values
            results = fixed_penalty_smooth_fdr(data, solver, _lambda, null_dist, signal_dist,
                                               verbose=max(0,verbose - 1),
                                               initial_values=initial_values)

            if verbose:
                print('Calculating degrees of freedom')

            plateaus = calc_plateaus(results['beta'], solver.edges)
            dof_trace[i] = len(plateaus)

            if verbose:
                print('Calculating AIC')

            # Get the negative log-likelihood
            log_likelihood_trace[i] = -_data_negative_log_likelihood(data, results['c'], null_dist, signal_dist)

            # Calculate AIC = 2k - 2ln(L)
            aic_trace[i] = 2. * dof_trace[i] - 2. * log_likelihood_trace[i]
            
            # Calculate AICc = AIC + 2k * (k+1) / (n - k - 1)
            aicc_trace[i] = aic_trace[i] + 2 * dof_trace[i] * (dof_trace[i]+1) / (data.shape[0] - dof_trace[i] - 1.)

            # Calculate BIC = -2ln(L) + k * (ln(n) - ln(2pi))
            bic_trace[i] = -2 * log_likelihood_trace[i] + dof_trace[i] * (np.log(len(data)) - np.log(2 * np.pi))

            # Track the best model thus far
            if best_idx is None or bic_trace[i] < bic_trace[best_idx]:
                best_idx = i
                best_plateaus = plateaus

            # Save the final run parameters to use for warm-starting the next iteration
            initial_values = results

            # Save the trace of all the resulting parameters
            beta_trace.append(results['beta'])
            w_trace.append(results['w'])
            c_trace.append(results['c'])

            if verbose:
                print('DoF: {0} AIC: {1} AICc: {2} BIC: {3}'.format(dof_trace[i], aic_trace[i], aicc_trace[i], bic_trace[i]))

        if verbose:
            print('Best setting (by BIC): lambda={0} [DoF: {1}, AIC: {2}, AICc: {3} BIC: {4}]'.format(lambda_grid[best_idx], dof_trace[best_idx], aic_trace[best_idx], aicc_trace[best_idx], bic_trace[best_idx]))

        return {'aic': aic_trace,
                'aicc': aicc_trace,
                'bic': bic_trace,
                'dof': dof_trace,
                'loglikelihood': log_likelihood_trace,
                'beta_iters': np.array(beta_trace),
                'posterior_iters': np.array(w_trace),
                'prior_iters': np.array(c_trace),
                'lambda_iters': lambda_grid,
                'best': best_idx,
                'betas': beta_trace[best_idx],
                'priors': c_trace[best_idx],
                'posteriors': w_trace[best_idx],
                'lambda': lambda_grid[best_idx],
                'plateaus': best_plateaus}

def fixed_penalty_smooth_fdr(data, solver, _lambda, null_dist, signal_dist, initial_values=None, verbose=0):
    converge = 1e-6
    max_steps = 30
    m_steps = 1
    m_converge = 1e-6

    w_iters = []
    beta_iters = []
    c_iters = []
    delta_iters = []

    delta = converge + 1
        
    if initial_values is None:
        beta = np.zeros(data.shape)
        prior_prob = np.exp(beta) / (1 + np.exp(beta))
    else:
        beta = initial_values['beta']
        prior_prob = initial_values['c']

    prev_nll = 0
    cur_step = 0
    
    while delta > converge and cur_step < max_steps:
        if verbose:
            print('Step #{0}'.format(cur_step))

        if verbose:
            print('\tE-step...')

        # Get the likelihood weights vector (E-step)
        post_prob = _e_step(data, prior_prob, null_dist, signal_dist)

        if verbose:
            print('\tM-step...')

        # Find beta using an alternating Taylor approximation and convex optimization (M-step)
        beta, initial_values = _m_step(beta, prior_prob, post_prob, _lambda,
                                       solver, m_converge, m_steps,
                                       max(0,verbose-1), initial_values)

        # Get the signal probabilities
        prior_prob = ilogit(beta)
        cur_nll = _data_negative_log_likelihood(data, prior_prob, null_dist, signal_dist)
        
        # Track the change in log-likelihood to see if we've converged
        delta = np.abs(cur_nll - prev_nll) / (prev_nll + converge)

        if verbose:
            print('\tDelta: {0}'.format(delta))

        # Track the step
        w_iters.append(post_prob)
        beta_iters.append(beta)
        c_iters.append(prior_prob)
        delta_iters.append(delta)

        # Increment the step counter
        cur_step += 1

        # Update the negative log-likelihood tracker
        prev_nll = cur_nll

        # DEBUGGING
        if verbose:
            print('\tbeta: [{0:.4f}, {1:.4f}]'.format(beta.min(), beta.max()))
            print('\tprior_prob:    [{0:.4f}, {1:.4f}]'.format(prior_prob.min(), prior_prob.max()))
            print('\tpost_prob:    [{0:.4f}, {1:.4f}]'.format(post_prob.min(), post_prob.max()))
            
    w_iters = np.array(w_iters)
    beta_iters = np.array(beta_iters)
    c_iters = np.array(c_iters)
    delta_iters = np.array(delta_iters)

    # Return the results of the run
    return {'beta': beta, 'w': post_prob, 'c': prior_prob,
            'z': initial_values['z'], 'u': initial_values['u'],
            'w_iters': w_iters, 'beta_iters': beta_iters,
            'c_iters': c_iters, 'delta_iters': delta_iters}

def _data_negative_log_likelihood(data, prior_prob, null_dist, signal_dist):
    '''Calculate the negative log-likelihood of the data given the weights.'''
    signal_weight = prior_prob * signal_dist.pdf(data)
    null_weight = (1-prior_prob) * null_dist.pdf(data)
    return -np.log(signal_weight + null_weight).sum()

def _e_step(data, prior_prob, null_dist, signal_dist):
    '''Calculate the complete-data sufficient statistics (weights vector).'''
    signal_weight = prior_prob * signal_dist.pdf(data)
    null_weight = (1-prior_prob) * null_dist.pdf(data)
    post_prob = signal_weight / (signal_weight + null_weight)
    return post_prob

def _m_step(beta, prior_prob, post_prob, _lambda,
                solver, converge, max_steps,
                verbose, initial_values):
    '''
    Alternating Second-order Taylor-series expansion about the current iterate
    '''
    prev_nll = _m_log_likelihood(post_prob, beta)
    delta = converge + 1
    cur_step = 0
    while delta > converge and cur_step < max_steps:
        if verbose:
            print('\t\tM-Step iteration #{0}'.format(cur_step))
            print('\t\tTaylor approximation...')

        # Cache the exponentiated beta
        exp_beta = np.exp(beta)

        # Form the parameters for our weighted least squares
        weights = (prior_prob * (1 - prior_prob))
        y = beta - (prior_prob - post_prob) / weights

        solver.set_values_only(y, weights=weights)
        if initial_values is None:
            initial_values = {'beta': solver.beta, 'z': solver.z, 'u': solver.u}
        else:
            solver.beta = initial_values['beta']
            solver.z = initial_values['z']
            solver.u = initial_values['u']
        solver.solve(_lambda)
        # if np.abs(beta).max() > 20:
        #     beta = np.clip(beta, -20, 20)
        #     u = None

        beta = initial_values['beta']

        # Get the current log-likelihood
        cur_nll = _m_log_likelihood(post_prob, beta)

        # Track the convergence
        delta = np.abs(prev_nll - cur_nll) / (prev_nll + converge)

        if verbose:
            print('\t\tM-step delta: {0}'.format(delta))

        # Increment the step counter
        cur_step += 1

        # Update the negative log-likelihood tracker
        prev_nll = cur_nll

    return beta, initial_values

def _m_log_likelihood(post_prob, beta):
    '''Calculate the log-likelihood of the betas given the weights and data.'''
    return (np.log(1 + np.exp(beta)) - post_prob * beta).sum()

def ilogit(x):
    return 1. / (1. + np.exp(-x))
