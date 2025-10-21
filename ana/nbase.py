#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#

import numpy as np
import os, logging, builtins
import itertools

try: 
    from hashlib import md5 
except ImportError: 
    from md5 import md5 



log = logging.getLogger(__name__) 

try:
    from scipy.stats import chi2 as _chi2
except ImportError:
    _chi2 = None 


def np_string(a):
    s = np.array2string(a, separator=",")
    return s.strip('[]').replace(' ', '')


def np_tostring(a, scale=1e6):
    b = a.copy()
    b[:,0] /= scale
    return " ".join(list(map(str,b.ravel()))) 

def np_fromstring(values, coldim=2, scale=1e6):
    a = np.fromstring(values, dtype=np.float32, sep=' ').reshape(-1, coldim) 
    a[:,0] *= scale     
    return a

np_fromtxt = lambda txt, dtype=np.float32:np.fromiter( map(float, txt.replace("("," ").replace(")"," ").replace(","," ").strip().split() ), dtype=dtype ) 



class Buf(np.ndarray): pass



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




def find_ranges(i):
    """  
    :param i: sorted list of integers

    E.g. given the set {0, 1, 2, 3, 4, 7, 8, 9, 11} I want to get { {0,4}, {7,9}, {11,11} }.

    http://stackoverflow.com/questions/4628333/converting-a-list-of-integers-into-range-in-python
    """
    func = lambda x,y:y-x

    for a, b in itertools.groupby(enumerate(i), func):
        b = list(b)
        yield b[0][1], b[-1][1]
    pass

def count_unique_truncating(vals):
    """  
    http://stackoverflow.com/questions/10741346/numpy-frequency-counts-for-unique-values-in-an-array
    """
    uniq = np.unique(vals)
    bins = uniq.searchsorted(vals)
    return np.vstack((uniq, np.bincount(bins))).T

def count_unique_old(vals):
    """
    """ 
    uniq = np.unique(vals)
    bins = uniq.searchsorted(vals)
    cnts = np.bincount(bins)
    return np.vstack((uniq, cnts.astype(np.uint64))).T

def count_unique_new(vals):
    """
    np.unique return_counts option requires at least numpy 1.9 
    """
    uniq, cnts = np.unique(vals, return_counts=True)
    return np.vstack((uniq, cnts.astype(np.uint64))).T 

def count_unique(vals):
    try:
        cu = count_unique_new(vals)
    except TypeError:
        cu = count_unique_old(vals)
    pass
    return cu 


def unique2D_subarray(a):
    """
    https://stackoverflow.com/questions/40674696/numpy-unique-2d-sub-array 
    """
    dtype1 = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))  #  eg for (10,4,4) -> np.dtype( (np.void, 
    b = np.ascontiguousarray(a.reshape(a.shape[0],-1)).view(dtype1)
    return a[np.unique(b, return_index=1)[1]]




def np_digest(a):
    """
    https://stackoverflow.com/questions/5386694/fast-way-to-hash-numpy-objects-for-caching

    file digest includes the header, not just the data : so will not match this 



    In [11]: md5(np.ascontiguousarray(f.hit[928534477].view(np.uint8))).hexdigest()
    Out[11]: 'ed3169da5f57135e2d1668acf2d0c525'


    """
    dig = md5()
    data = np.ascontiguousarray(a.view(np.uint8))
    dig.update(data)
    return dig.hexdigest()


def array_digest(a):
    """
    https://stackoverflow.com/questions/5386694/fast-way-to-hash-numpy-objects-for-caching

    file digest includes the header, not just the data : so will not match this 
    """
    return np_digest(a)


def cus(s):
    """
    :param s: array of shape (n,)
    :return cus: count unique sorted array of shape (num_unique, 2)
    """
    u,c = np.unique(s, return_counts=True) 
    o = np.argsort(c)[::-1]
    cu = np.zeros( (len(u),2), dtype=np.uint64 )
    cu[:,0] = u[o]
    cu[:,1] = c[o]
    return cu 




def count_unique_sorted(vals):
    """
    Older numpy has problem with the argsort line when cu is empty
    """
    #vals = vals.astype(np.uint64)  cannot do this with -ve pdgcode
    cu = count_unique(vals) 
    if len(cu) != 0:
        cu = cu[np.argsort(cu[:,1])[::-1]]  # descending frequency order
    pass
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

    * https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html

    For ndf:10 you want to know if a chi2 value of 18.307
    should be regarded as consistent with a null hypothesis.
    By looking at the probability that the chi2 where more than 
    that::

        In [2]: from scipy.stats import chi2 as _chi2
        In [27]: 1 - _chi2.cdf(18.307, 10)  ## chi2 of 18.307 for ndf:10 : on the edge of being significant deviation
        Out[27]: 0.05000058909139815

        In [4]: 1-_chi2.cdf(20, 10 )   ## chi2 of 20 for ndf:10 : regarded as significant deviation from null 
        Out[4]: 0.02925268807696113

    A p-value of less than or equal to 0.05 is regarded as evidence of a
    statistically significant result, and in these cases, the null hypothesis
    should be rejected in favor of the alternative hypothesis.


    # probability of getting a chi2 <= c2obs for the ndf 

    # https://onlinecourses.science.psu.edu/stat414/node/147

    :google:`interpret chi2 values`

    ::


        from scipy.stats import chi2 as _chi2

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
    deviations, so I want the p-value to be close to 1 indicating nothing of significance.

    ::

        In [56]: _chi2.cdf( [3.94,4.87,6.18,7.27,9.34,11.78,13.44,15.99,18.31,23.21,29.59], 10 )
        Out[56]: array([ 0.05  ,  0.1003,  0.2001,  0.3003,  0.4998,  0.6999,  0.7999,  0.9001,  0.95  ,  0.99  ,  0.999 ])
        ## probability of a chi2 value less than those values for ndf 10 

        In [57]: 1 - _chi2.cdf( [3.94,4.87,6.18,7.27,9.34,11.78,13.44,15.99,18.31,23.21,29.59], 10 )   ## P-value (Probability)
        Out[57]: array([ 0.95  ,  0.8997,  0.7999,  0.6997,  0.5002,  0.3001,  0.2001,  0.0999,  0.05  ,  0.01  ,  0.001 ])
        ## probability of a chi2 greater than those values for ndf 10 


    * ~/opticks_refs/PoissonConsistency.pdf
    * http://www.hep.caltech.edu/~fcp/statistics/hypothesisTest/PoissonConsistency/PoissonConsistency.pdf 

    * https://www.scribbr.com/statistics/chi-square-distributions/
    * https://www.scribbr.com/statistics/chi-square-tests/
    * https://www.scribbr.com/statistics/chi-square-distribution-table/
    * https://www.scribbr.com/wp-content/uploads/2022/05/Chi-square-table.pdf

    In [20]: df = np.arange(1,31)

    ## so called 5% critical values : to judge null hypotheses for different 
    ## degrees of freedom 
    ## Alternatively 

    In [22]: df = np.arange(1,31)

    In [23]: cr5 = _chi2.ppf( 0.95,df)

    In [24]: _chi2.cdf( cr5 , df)
    Out[24]: array([0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95])

    In [25]: np.c_[df,cr5]
    Out[25]: 
    array([[ 1.   ,  3.841],
           [ 2.   ,  5.991],
           [ 3.   ,  7.815],
           [ 4.   ,  9.488],
           [ 5.   , 11.07 ],
           [ 6.   , 12.592],
           [ 7.   , 14.067],
           [ 8.   , 15.507],
           [ 9.   , 16.919],
           [10.   , 18.307],
           [11.   , 19.675],
           [12.   , 21.026],
           [13.   , 22.362],
           [14.   , 23.685],
           [15.   , 24.996],
           [16.   , 26.296],
           [17.   , 27.587],


    * https://www.nevis.columbia.edu/~seligman/root-class/html/appendix/statistics/ChiSquaredDOF.html

    Chi2 critical values

    * https://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm

    """
    if _chi2 is None:
        return 1

    p_value = 1 - _chi2.cdf(c2obs, ndf) 
    return p_value


def chi2one(a, b):
    return (a-b)*(a-b)/(a+b)

def chi2(a_, b_, cut=30):
    """
    :param a_: array of counts
    :param b_: array of counts
    :param cut: applied to sum of and and b excludes low counts from the chi2 
    :return c2,c2n,c2c:

    *c2*
         array with elementwise square of difference over the sum 
         (masked by the cut on the sum, zeros provided for low stat entries)
    *c2n*
         number of bins within the count cut
    *c2c*
         number of bins not within count cut 
  
    ::

        c2, c2n, c2c = chi2(a, b)
        c2ndf = c2.sum()/c2n

    # ChiSquared or KS
    # http://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm 
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty
    # http://stats.stackexchange.com/questions/7400/how-to-assess-the-similarity-of-two-histograms


    # http://www.hep.caltech.edu/~fcp/statistics/hypothesisTest/PoissonConsistency/PoissonConsistency.pdf
    # ~/opticks_refs/PoissonConsistency.pdf   Link now dead ?    
    # "Testing Consistency of Two Histograms", Frank Porter, California Institute of Technology

    """
    a = a_.astype(np.float32)
    b = b_.astype(np.float32)

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
    pass
    log.info("test_count_unique_ %16x %16x %s  %s %s  " % (a, a_, msg, a, a_) )


def test_count_unique():

    log.info("test_count_unique")
    vals=[0xfedcba9876543210,0xffffffffffffffff]

    msk_ = lambda n:(1 << 4*n) - 1 
    for n in range(16):
        msk = msk_(n)    
        vals.append(msk)

    for fn in [count_unique_new, count_unique_old, count_unique_truncating]:
        log.info(fn.__name__)
        map(lambda v:test_count_unique_(fn, v), vals)


def test_chi2():
    a = np.array([0,100,500,500],dtype=np.int32)
    b = np.array([0,100,500,0],  dtype=np.int32)

    c2,c2n,c2c = chi2(a,b, cut=30)


def test_count_unique_2():
    log.info("test_count_unique_2")
    t = np.array( [1,2,2,3,3,3,4,4,4,4,5,5,5,5,5], dtype=np.uint32 )
    x = np.array( [[1,1],[2,2],[3,3],[4,4], [5,5]], dtype=np.uint64 )

    r1 = count_unique( t )
    assert np.all( r1 == x )


def test_count_unique_sorted():
    log.info("test_count_unique_sorted")
    t = np.array( [1,2,2,3,3,3,4,4,4,4,5,5,5,5,5], dtype=np.uint32 )
    y = np.array( [[5,5],[4,4],[3,3],[2,2], [1,1]], dtype=np.uint64 )

    r = count_unique_sorted( t )
    assert np.all( r == y )


def test_count_unique_sorted_empty():
    log.info("test_count_unique_sorted_empty np.__version__ %s " % np.__version__ )
    t = np.array( [], dtype=np.uint32 )
    r = count_unique_sorted( t )

    u = np.array( [], dtype=np.uint64 )
    c = np.array( [], dtype=np.uint64 )
    x = np.vstack( (u,c) ).T 

    assert np.all( r == x )



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    log.info("np.__version__ %s " % np.__version__ )

    #test_count_unique()
    #test_chi2()
    test_count_unique_2()
    test_count_unique_sorted()
    test_count_unique_sorted_empty()



  



