#!/usr/bin/env python
import numpy as np
import os, logging
log = logging.getLogger(__name__) 

try:
    from scipy.stats import chi2 as _chi2
except ImportError:
    _chi2 = None 


def ahash(a):
    """
    * http://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array    

    ::

        hash(a.data)
        Out[7]: 7079931724019902235

        In [8]: "%x" % hash(a.data)
        Out[8]: '6240f8645439a71b'

    """
    a.flags.writeable = False
    return "%x" % hash(a.data)


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

costheta_ = lambda a,b:np.sum(a * b, axis = 1)/(vnorm(a)*vnorm(b))


def chi2_pvalue( c2obs, ndf ):
    """
    :param c2obs: observed chi2 value
    :param ndf: 
    :return pvalue:  1 - _chi2.cdf(c2obs, ndf) 

    # probability of getting a chi2 <= c2obs for the ndf 

    # https://onlinecourses.science.psu.edu/stat414/node/147

    ::

        In [49]: _chi2.cdf( 15.99, 10 )   ## for ndf 10, probability for chi2 < 15.99 is 0.900
        Out[49]: 0.90008098002000925

        In [53]: _chi2.cdf( range(10,21), 10 )    ## probability of getting chi2 < the value
        Out[53]: array([ 0.5595,  0.6425,  0.7149,  0.7763,  0.827 ,  0.8679,  0.9004,  0.9256,  0.945 ,  0.9597,  0.9707])

        In [54]: 1 - _chi2.cdf( range(10,21), 10 )  ## probability of getting chi2 > the value
        Out[54]: array([ 0.4405,  0.3575,  0.2851,  0.2237,  0.173 ,  0.1321,  0.0996,  0.0744,  0.055 ,  0.0403,  0.0293])


    * https://en.wikipedia.org/wiki/Chi-squared_distribution

    The p-value is the probability of observing a test statistic at least as
    extreme in a chi-squared distribution. Accordingly, since the cumulative
    distribution function (CDF) for the appropriate degrees of freedom (df) gives
    the probability of having obtained a value less extreme than this point,
    subtracting the CDF value from 1 gives the p-value. The table below gives a
    number of p-values matching to chi2 for the first 10 degrees of freedom.

    A low p-value indicates greater statistical significance, i.e. greater
    confidence that the observed deviation from the null hypothesis is significant.
    A p-value of 0.05 is often used as a cutoff between significant and
    not-significant results.

    Of course for Opticks-CFG4 comparisons I wish to see no significant 
    deviations, so I want the p-value to be large indicating nothing of significance.

    ::

        In [56]: _chi2.cdf( [3.94,4.87,6.18,7.27,9.34,11.78,13.44,15.99,18.31,23.21,29.59], 10 )
        Out[56]: array([ 0.05  ,  0.1003,  0.2001,  0.3003,  0.4998,  0.6999,  0.7999,  0.9001,  0.95  ,  0.99  ,  0.999 ])

        In [57]: 1 - _chi2.cdf( [3.94,4.87,6.18,7.27,9.34,11.78,13.44,15.99,18.31,23.21,29.59], 10 )   ## P-value (Probability)
        Out[57]: array([ 0.95  ,  0.8997,  0.7999,  0.6997,  0.5002,  0.3001,  0.2001,  0.0999,  0.05  ,  0.01  ,  0.001 ])


    """
    if _chi2 is None:
        return None 

    p_value = 1 - _chi2.cdf(c2obs, ndf) 
    return p_value



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



  



