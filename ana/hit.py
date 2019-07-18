#!/usr/bin/env python
"""

hit.py
========

Checking relationship betweet hit counts and photon counts 
from a photon scan.  The "hits" are not real ones, just some 
photon flag mask chosen to give some statistics for machinery testing. 

From 1M up to 60M get a very uniform gradient, from 70M continue
with the same gradient but with a great big offset, as if suddenly
lost around 67M photons. 

Sawtooth plot::

    In [9]: np.unique(n, axis=0)
    Out[9]: 
    array([[  1000000,       516],
           [ 10000000,      5212],
           [ 20000000,     10406],
           [ 30000000,     15675],
           [ 40000000,     20802],
           [ 50000000,     26031],
           [ 60000000,     31126],
           [ 70000000,      1562],    ## glitch down to a hit count would expect from around 3M photons
           [ 80000000,      6663],
           [ 90000000,     11963],
           [100000000,     17122]], dtype=int32)


    In [18]:  u[:,0]/u[:,1]
    Out[18]: 
    array([ 1937,  1918,  1921,  1913,  1922,  1920,  1927, 44814, 12006,
            7523,  5840], dtype=int32)

    In [20]:  u[:7,0]/u[:7,1]
    Out[20]: array([1937, 1918, 1921, 1913, 1922, 1920, 1927], dtype=int32)

    In [21]: np.average( u[:7,0]/u[:7,1])
    Out[21]: 1922.5714285714287

    ## gradient same beyond the glitch
 
    In [37]: (u[8:,0]-u[7,0])/(u[8:,1]-u[7,1])
    Out[37]: array([1960, 1922, 1928], dtype=int32)


"""
import os, sys, re, logging, numpy as np
from collections import OrderedDict as odict

from opticks.ana.num import Num
from opticks.ana.base import findfile

import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


class Arr(object):

    @classmethod
    def Find(cls, pfx, name, qcat=None):
        base = os.path.expandvars(os.path.join("$TMP", pfx ))
        rpaths = findfile(base, name ) 
        aa = []
        for rp in rpaths:
            path = os.path.join(base, rp) 
            elem = path.split("/")
            cat = elem[elem.index("evt")+1]
            if not qcat is None and not cat.startswith(qcat): continue  
            snpho = cat.split("_")[-1]
            npho = Num.Int(snpho)
            aa.append(Arr(path,cat, npho))
        pass
        return aa
     
    def __init__(self, path, cat, npho):
        self.path = path
        self.cat = cat  
        self.npho = npho
        self.a = np.load(path)
        self.items = len(self.a)


if __name__ == '__main__':


    np.set_printoptions(suppress=True, precision=3)

    plt.ion()

    aa = Arr.Find("scan-ph", "ht.npy", "cvd_1_rtx_1_" )
    print(" aa %s " % len(aa))     
       
    n = np.zeros( [len(aa),2] , dtype=np.int32 ) 
    for i,a in enumerate(sorted(aa, key=lambda a:a.npho)):
        n[i,0] = a.npho
        n[i,1] = a.items
    pass
    print(n)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot( n[:,0], n[:,1], "o--" )

    plt.show() 

    


       

