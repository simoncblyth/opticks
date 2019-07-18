scan-ph-hits-not-scaling
============================

Context
---------

* :doc:`tboolean-generateoverride-photon-scanning`


mini scan with hit saving
-----------------------------

::

    [blyth@localhost torch]$ scan-ph-
    ts box --pfx scan-ph --cat cvd_1_rtx_1_1M --generateoverride 1000000 --compute --production --savehit --multievent 10 --nog4propagate --rngmax 3 --cvd 1 --rtx 1
    ts box --pfx scan-ph --cat cvd_1_rtx_1_10M --generateoverride 10000000 --compute --production --savehit --multievent 10 --nog4propagate --rngmax 10 --cvd 1 --rtx 1
    ts box --pfx scan-ph --cat cvd_1_rtx_1_100M --generateoverride 100000000 --compute --production --savehit --multievent 10 --nog4propagate --rngmax 100 --cvd 1 --rtx 1


Something wrong from 70M onwards, not getting expected number of hits::

    [blyth@localhost evt]$ np.py | grep 10/ht.npy
    . :                             ./cvd_1_rtx_1_1M/torch/10/ht.npy :          (516, 4, 4) : 4b3b0a15df483aea004f476ca2552647 : 20190717-1658 
    . :                            ./cvd_1_rtx_1_10M/torch/10/ht.npy :         (5212, 4, 4) : 613acc8a11a69a3266e8005740e10a7d : 20190717-1659 
    . :                            ./cvd_1_rtx_1_20M/torch/10/ht.npy :        (10406, 4, 4) : 13ff832c1b5f89145e47bba14f0bf689 : 20190717-1700 
    . :                            ./cvd_1_rtx_1_30M/torch/10/ht.npy :        (15675, 4, 4) : ef7eeea8c00ccf69285e3a4d79adb277 : 20190717-1701 
    . :                            ./cvd_1_rtx_1_40M/torch/10/ht.npy :        (20802, 4, 4) : 7d48e622d1ff23d62746c082de1a7758 : 20190717-1702 
    . :                            ./cvd_1_rtx_1_50M/torch/10/ht.npy :        (26031, 4, 4) : cd3d98da484f73d0a6548877467843ee : 20190717-1704 
    . :                            ./cvd_1_rtx_1_60M/torch/10/ht.npy :        (31126, 4, 4) : 24544e474794897b94d0282158636042 : 20190717-1706 
    . :                            ./cvd_1_rtx_1_70M/torch/10/ht.npy :         (1562, 4, 4) : 6a852d32c1867d53fc922efca5362a75 : 20190717-1708 
    . :                            ./cvd_1_rtx_1_80M/torch/10/ht.npy :         (6663, 4, 4) : d31dc0e7829bce284288ac1b37c9cb42 : 20190717-1710 
    . :                            ./cvd_1_rtx_1_90M/torch/10/ht.npy :        (11963, 4, 4) : 153a9e77f9d893f34962d3082a122503 : 20190717-1713 
    . :                           ./cvd_1_rtx_1_100M/torch/10/ht.npy :        (17122, 4, 4) : c373727993f3e681953993d479a0e0a6 : 20190717-1715 
    . :                             ./cvd_1_rtx_0_1M/torch/10/ht.npy :          (516, 4, 4) : 38ecf5480303c99589f98e9c0f12700b : 20190717-1632 


hit.py
--------

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



