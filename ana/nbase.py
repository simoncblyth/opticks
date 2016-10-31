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

def vnorm(a):
    """
    Older numpy lacks the third axis argument form of np.linalg.norm 
    so this replaces it::

        #r = np.linalg.norm(xyz[:,:2], 2, 1)  
        r = vnorm(xyz[:,:2]) 

    """
    return np.sqrt(np.sum(a*a,1))


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

def ratio(a, b):
    m = np.logical_and(a > 0, b > 0)

    ab = np.zeros((len(a),2))
    ba = np.zeros((len(a),2))

    ab[m,0] = a[m]/b[m] 
    ab[m,1] = np.sqrt(a[m])/b[m] 

    ba[m,0] = b[m]/a[m]
    ba[m,1] = np.sqrt(b[m])/a[m]

    return ab, ba


def decompression_bins(cbins, *vals):
    """
    :param cbins: full range decompressed bins 
    :param vals:

    Compression can be considered to be a very early (primordial) binning.  
    To avoid artifacts all subsequent binning needs to be
    use bins that correspond to these originals. 

    This provides a subset of full range decompression bins, 
    corresponding to a value range.

    ::

        In [3]: cbins = cf.a.pbins()

        In [4]: cbins
        Out[4]: array([ -24230.625 ,  -24242.3772,  -24254.1294, ..., -794375.8706, -794387.6228, -794399.375 ])

        In [15]: np.where( cbins <=  -60254.1294 )
        Out[15]: (array([ 3066,  3067,  3068, ..., 65532, 65533, 65534]),)


    """
    vmin = min(map(lambda _:_.min(), vals))
    vmax = max(map(lambda _:_.max(), vals))
    width = (vmax - vmin)
    widen = width*0.1

    vmin = vmin-widen
    vmax = vmax+widen

    try:
        imin = np.where(cbins>=vmin)[0][0]
        imax = np.where(cbins<=vmax)[0][-1]
        inum = imax - imin  
    except IndexError:
        log.warning(" MISMATCH BETWEEN COMPRESSION BINS AND VALUES cbins %s vmin %s vmax %s width %s " % (repr(cbins), vmin, vmax, width))
        inum = None

    if inum is None:
       return None

    pass
    if inum == 0:
        log.warning("special case handling of all values the same")
        bins = np.linspace(vmin-1,vmax+1,3)
    else:
        bins = np.linspace(vmin,vmax,inum)
    pass
    return bins

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    cbins = np.linspace(-300,300,10)
    avals = np.repeat(300,1000)
    bvals = np.repeat(300,1000)

    rbins = decompression_bins(cbins, avals, bvals)





