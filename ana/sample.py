#!/usr/bin/env python
"""

https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere

"""
import numpy as np

def sample_linear(n, lo=-1., hi=1.):
    """
    :param n: number of values
    :param lo: min value
    :param hi: max value
    :return ll: array of shape (n,) with random sample values uniformly between lo and hi
    """
    mi = (lo + hi)/2.
    ex = (hi - lo)/2.
    uu = 2.*np.random.rand(n) - 1.   # -1->1 
    ll = mi + uu*ex      
    return ll 

def sample_linspace(n, lo=-1., hi=1. ):
    return np.linspace(lo, hi, n ) 

def sample_trig(n):
    """
    :param n: number of points 
    """
    theta = 2*np.pi*np.random.rand(n)
    phi = np.arccos(2*np.random.rand(n)-1)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.array([x,y,z])

def sample_normals(npoints):
    vec = np.random.randn(3, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def sample_reject(npoints):
    vec = np.zeros((3,npoints))
    abc = 2*np.random.rand(3,npoints)-1
    norms = np.linalg.norm(abc,axis=0) 
    mymask = norms<=1
    abc = abc[:,mymask]/norms[mymask]
    k = abc.shape[1]
    vec[:,0:k] = abc
    while k<npoints:
       abc = 2*np.random.rand(3)-1
       norm = np.linalg.norm(abc)
       if 1e-5 <= norm <= 1:  
           vec[:,k] = abc/norm
           k = k+1
    return vec


if __name__ == '__main__':

    n = 10 
    a = sample_trig(n)
    b = sample_normals(n)
    c = sample_reject(n)


