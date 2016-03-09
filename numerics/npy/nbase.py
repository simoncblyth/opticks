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

    # ChiSquared or KS
    # http://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm 
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty
    # http://stats.stackexchange.com/questions/7400/how-to-assess-the-similarity-of-two-histograms
    # http://www.hep.caltech.edu/~fcp/statistics/hypothesisTest/PoissonConsistency/PoissonConsistency.pdf
    """
    msk = a+b > cut
    c2 = np.zeros_like(a)
    c2[msk] = np.power(a-b,2)[msk]/(a+b)[msk]
    return c2, len(a[msk]), len(a[~msk]) 
 

def decompression_bins(cbins, *vals):
    """
    :param cbins: full range decompressed bins 
    :param vals:

    Compression can be considered to be a very early (primordial) binning.  
    To avoid artifacts all subsequent binning needs to be
    use bins that correspond to these originals. 

    This provides a subset of full range decompression bins, 
    corresponding to a value range.
    """
    vmin = min(map(lambda _:_.min(), vals))
    vmax = max(map(lambda _:_.max(), vals))
    width = (vmax - vmin)
    widen = width*0.1

    vmin = vmin-widen
    vmax = vmax+widen
    imin = np.where(cbins>=vmin)[0][0]
    imax = np.where(cbins<=vmax)[0][-1]
    pass
    inum = imax - imin  
    if inum == 0:
        log.warning("special case handling of all values the same")
        bins = np.linspace(vmin-1,vmax+1,3)
    else:
        bins = np.linspace(vmin,vmax,inum)
    return bins

if __name__ == '__main__':

    cbins = np.linspace(-300,300,10)
    avals = np.repeat(300,1000)
    bvals = np.repeat(300,1000)

    rbins = decompression_bins(cbins, avals, bvals)





