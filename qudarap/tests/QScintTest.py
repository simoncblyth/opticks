#!/usr/bin/env python
"""
QScintTest.py
==================

::

    ipython -i tests/QScintTest.py 

"""
import os, sys, logging, numpy as np
import matplotlib.pyplot as plt
from opticks.ana.nload import np_load
from opticks.ana.key import keydir

log = logging.getLogger(__name__)

if __name__ == '__main__':

    ok = os.environ["OPTICKS_KEY"]
    kd = keydir(ok)
    aa = np_load(os.path.join(kd,"GScintillatorLib/GScintillatorLib.npy"))
    fc = np_load(os.path.join(kd,"GScintillatorLib/LS/FASTCOMPONENT.npy"))
    sc = np_load(os.path.join(kd,"GScintillatorLib/LS/SLOWCOMPONENT.npy"))

    print("aa:%s" % str(aa.shape))
    print("fc:%s" % str(fc.shape))
    print("sc:%s" % str(sc.shape))
    assert aa.shape == (1, 4096, 1)
    assert np.all( fc == sc )

    fold = "/tmp/QScintTest" 
    w = np.load(os.path.join(fold, "wavelength.npy"))


    p = np.load(os.path.join(fold, "photon.npy"))
    dir = p[:,1,:3]
    pol = p[:,2,:3]
    dirpol = (dir*pol).sum(axis=1)   # check transverse pol 
    dirpol_max = np.abs(dirpol).max()
    assert dirpol_max < 1e-6, dirpol_max   









    plt.ion()

    wd = np.linspace(60,820,256) - 1.  
    # reduce bin edges by 1nm to avoid aliasing artifact in the histogram

    mid = (wd[:-1]+wd[1:])/2.     # bin middle

    counts, edges = np.histogram(w, bins=wd )
    fcounts = counts.astype(np.float32)
    fcounts  /= fcounts.sum()
    fcounts  /= 2.98   # bin width nm

    plt.close()

    plt.plot( edges[:-1], fcounts, drawstyle="steps-mid")


    _fc = fc[:,1]
    _fc /= _fc.sum()
    _fc /= 20.       # bin width nm

    plt.plot( fc[:,0], _fc ) 

    #plt.plot( mid,  pl ) 
    
    plt.axis( [w.min() - 100, w.max() + 100, 0, fcounts.max()*1.1 ])  

    #plt.hist(w, bins=256)   # 256 is number of unique wavelengths (from record compression)

