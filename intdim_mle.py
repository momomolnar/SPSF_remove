"""
Implementation of 'Maximum Likelihood Estimation of Intrinsic Dimension' by Elizaveta Levina and Peter J. Bickel
how to use
----------
The goal is to estimate intrinsic dimensionality of data, the estimation of dimensionality is scale dependent
(depending on how much you zoom into the data distribution you can find different dimesionality), so they
propose to average it over different scales, the interval of the scales [k1, k2] are the only parameters of the algorithm.
This code also provides a way to repeat the estimation with bootstrapping to estimate uncertainty.
Here is one example with swiss roll :
from sklearn.datasets import make_swiss_roll
X, _ = make_swiss_roll(1000)
k1 = 10 # start of interval(included)
k2 = 20 # end of interval(included)
intdim_k_repeated = repeated(intrinsic_dim_scale_interval,
                             X,
                             mode='bootstrap',
                             nb_iter=500, # nb_iter for bootstrapping
                             verbose=1,
                             k1=k1, k2=k2)
intdim_k_repeated = np.array(intdim_k_repeated)
# the shape of intdim_k_repeated is (nb_iter, size_of_interval) where
# nb_iter is number of bootstrap iterations (here 500) and size_of_interval
# is (k2 - k1 + 1).
# Plotting the histogram of intrinsic dimensionality estimations repeated over
# nb_iter experiments
plt.hist(intdim_k_repeated.mean(axis=1))
"""
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def intrinsic_dim_sample_wise_old(X, k=5):
    neighb = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, ind = neighb.kneighbors(X)
    dist = dist[:, 1:]
    dist = dist[:, 0:k]
    assert dist.shape == (X.shape[0], k)
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k-1])
    d = d.sum(axis=1) / (k - 2)
    d = 1. / d
    intdim_sample = d
    return intdim_sample


def intrinsic_dim_sample_wise_parallel(X, k=5):
    neighb = NearestNeighbors(n_neighbors=(k + 1)).fit(X)
    dist, ind = neighb.kneighbors(X)
    dist = dist[:, 1:]
    dist = dist[:, 0:k]
    assert dist.shape == (X.shape[0], k)
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k-1])
    d = d.sum(axis=1) / (k - 1)
    # print(f'd shape is {d.shape}')
    assert d.shape == (X.shape[0],)
    d = np.mean(d)
    d = 1. / d
    intdim_sample = d
    return intdim_sample


def intrinsic_dim_sample_wise(X, k=5):
    neighb = NearestNeighbors(n_neighbors=(k + 1)).fit(X)
    dist, ind = neighb.kneighbors(X)
    dist = dist[:, 1:]
    dist = dist[:, 0:k]
    assert dist.shape == (X.shape[0], k)
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k-1])
    assert d.shape == (X.shape[0], (k-1))
    d = d.sum(axis=1) / (k - 1)
    # d = 1. / d
    intdim_sample = d
    return intdim_sample


def test_dim_parallel():
    test_data = np.random.rand(1000, 100)
    intdim = []
    for ii in range(5, 40, 2):
        intdim.append(intrinsic_dim_sample_wise_parallel(test_data,
                                                         k=ii))
    intdim = np.array(intdim)
    plt.plot(intdim)
    plt.show()


def test_dim_parallel_real_data():
    a = np.load('CaProfiles.npz')
    caData = np.reshape(a['caData'], (25000, 130))
    caData_s = caData[3000:24000:5, :]
    intdim = []
    for ii in range(5, 40, 2):
        intdim.append(intrinsic_dim_sample_wise_parallel(caData_s,
                                                         k=ii))
    intdim = np.array(intdim)
    plt.plot(intdim)
    plt.show()



def intrinsic_dim_scale_interval_logscale(X, k1=10, k2=20, numSamples=100):
    # remove duplicates in case you use bootstrapping
    X = pd.DataFrame(X).drop_duplicates().values
    intdim_k = []
    k_values = np.logspace(np.log10(k1), np.log10(k2), num=numSamples)
    for k in k_values:
        # I think this should be
        # changed to the mean of the inverses
        m = intrinsic_dim_sample_wise(X, k).mean()
        intdim_k.append(1/m)
    return intdim_k


def intrinsic_dim_scale_interval(X, k1=10, k2=20):
    # remove duplicates in case you use bootstrapping
    X = pd.DataFrame(X).drop_duplicates().values
    intdim_k = []
    for k in range(k1, k2 + 1):
        # I think this should be
        # changed to the mean of the inverses
        m = intrinsic_dim_sample_wise(X, k).mean()
        intdim_k.append(1/m)
    return intdim_k


def repeated(func, X, nb_iter=100, random_state=None, verbose=0,
             mode='bootstrap', **func_kw):
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)
    nb_examples = X.shape[0]
    results = []

    iters = range(nb_iter)
    if verbose > 0:
        iters = tqdm(iters)
    for i in iters:
        if mode == 'bootstrap':
            Xr = X[rng.randint(0, nb_examples, size=nb_examples)]
        elif mode == 'shuffle':
            ind = np.arange(nb_examples)
            rng.shuffle(ind)
            Xr = X[ind]
        elif mode == 'same':
            Xr = X
        else:
            raise ValueError('unknown mode : {}'.format(mode))
        results.append(func(Xr, **func_kw))
    return results


if __name__ == "__main__":
    test_dim_parallel()
    test_dim_parallel_real_data()
