#!/usr/bin/env python
import numpy as np
import os, logging
log = logging.getLogger(__name__) 


def count_unique(vals):
    """  
    http://stackoverflow.com/questions/10741346/numpy-frequency-counts-for-unique-values-in-an-array
    """
    uniq = np.unique(vals)
    bins = uniq.searchsorted(vals)
    return np.vstack((uniq, np.bincount(bins))).T

def count_unique_sorted(vals):
    vals = vals.astype(np.uint64)
    cu = count_unique(vals)
    cu = cu[np.argsort(cu[:,1])[::-1]]  # descending frequency order
    return cu.astype(np.uint64)

def chi2(a, b, cut=30):
    """
    ::

        c2, c2n = chi2(a, b)
        c2ndf = c2.sum()/c2n

    """
    msk = a+b > cut
    c2 = np.zeros_like(a)
    c2[msk] = np.power(a-b,2)[msk]/(a+b)[msk]
    return c2, len(a[msk]) 
 


