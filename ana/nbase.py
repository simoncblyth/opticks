#!/usr/bin/env python
import numpy as np
import os, logging
log = logging.getLogger(__name__) 


def count_unique_truncating(vals):
    """  
    http://stackoverflow.com/questions/10741346/numpy-frequency-counts-for-unique-values-in-an-array
    """
    uniq = np.unique(vals)
    bins = uniq.searchsorted(vals)
    return np.vstack((uniq, np.bincount(bins))).T

def count_unique_old(vals):
    uniq = np.unique(vals)
    bins = uniq.searchsorted(vals)
    cnts = np.bincount(bins)
    return np.vstack((uniq, cnts.astype(np.uint64))).T

def count_unique(vals):
    """
    np.unique return_counts option requires at lease numpy 1.9 
    """
    uniq, cnts = np.unique(vals, return_counts=True)
    return np.vstack((uniq, cnts.astype(np.uint64))).T 


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

vnorm_ = lambda _:np.sqrt(np.sum(_*_,1))


def chi2(a, b, cut=30):
    """
    :param a: array of counts
    :param b: array of counts
    :param cut: applied to sum of and and b excludes low counts from the chi2 
    :return c2,c2n,c2c:

    *c2*
         array with elementwise square of difference over the sum 
         (masked by the cut on the sum, zeros provided for low stat entries)
    *c2n*
         number of counts within the mask
    *c2c*
         number of counts not within the mask
  
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

    c2n = len(a[msk])
    c2c = len(a[~msk]) 

    assert c2n + c2c == len(a)

    return c2, c2n, c2c

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




def test_decompression_bins():
    cbins = np.linspace(-300,300,10)
    avals = np.repeat(300,1000)
    bvals = np.repeat(300,1000)

    rbins = decompression_bins(cbins, avals, bvals)




def test_count_unique_(fn, a):
    """
    count_unique appears to go via floats which 
    looses precision for large numbers
    """

    aa = np.array([ a,a,a ], dtype=np.uint64 )

    cu = fn(aa)
    n = cu[0,1]
    assert n == 3

    a_ = np.uint64(cu[0,0])

    ok = a_ == a

    if ok:
         msg = "OK" 
    else:
         msg = "FAIL"


    log.info("test_count_unique_ %16x %16x %s  %s %s  " % (a, a_, msg, a, a_) )

        


def test_count_unique():

    vals=[0xfedcba9876543210,0xffffffffffffffff]

    msk_ = lambda n:(1 << 4*n) - 1 
    for n in range(16):
        msk = msk_(n)    
        vals.append(msk)

    for fn in [count_unique, count_unique_old, count_unique_truncating]:
        log.info(fn.__name__)
        map(lambda v:test_count_unique_(fn, v), vals)


def test_chi2():
    a = np.array([0,100,500,500],dtype=np.int32)
    b = np.array([0,100,500,0],  dtype=np.int32)

    c2,c2n,c2c = chi2(a,b, cut=30)







if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    #test_count_unique()
    #test_chi2()
    a = np.array([0,100,500,500],dtype=np.int32)
    b = np.array([0,100,500,0],  dtype=np.int32)

    c2,c2n,c2c = chi2(a,b, cut=30)



