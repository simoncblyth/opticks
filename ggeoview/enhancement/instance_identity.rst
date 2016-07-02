Instance Identity
===================

OptiX geometry instances (each with a different transform associated) 
need to have separate identity, so can know which PMT gets hit for example.

DYB GMergedMesh/1, the 5 solids of the PMT::

    In [1]: s = np.load("sensors.npy")

    In [3]: s.shape
    Out[3]: (2928, 1)

    In [3]: s.shape
    Out[3]: (2928, 1)

    In [4]: s[:,0]
    Out[4]: array([1, 1, 1, ..., 5, 5, 5], dtype=int32)

    In [5]: count_unique(s[:,0])
    Out[5]: 
    array([[  1, 720],
           [  2, 672],
           [  3, 960],
           [  4, 480],
           [  5,  96]])

::


    In [7]: n = np.load("nodeinfo.npy")

    In [8]: n
    Out[8]: 
    array([[ 720,  362, 3199, 3155],
           [ 672,  338, 3200, 3199],
           [ 960,  482, 3201, 3200],
           [ 480,  242, 3202, 3200],
           [  96,   50, 3203, 3200]], dtype=uint32)


Observations:

* nodeinfo just covers the 1st instance. 
* only the 3rd solid, the cathode, is really a sensor 



Using the instance identity buffer to handle this::

    In [1]: ii = np.load("iidentity.npy")

    In [2]: ii.shape
    Out[2]: (3360, 4)

    In [3]: ii.reshape(-1,5,4).shape
    Out[3]: (672, 5, 4)

    In [4]: ii.reshape(-1,5,4)
    Out[4]: 
    array([[[ 3199,    47,    19,     0],
            [ 3200,    46,    20,     0],
            [ 3201,    43,    21,     3],
            [ 3202,    44,     1,     0],
            [ 3203,    45,     1,     0]],

           [[ 3205,    47,    19,     0],
            [ 3206,    46,    20,     0],
            [ 3207,    43,    21,     8],
            [ 3208,    44,     1,     0],
            [ 3209,    45,     1,     0]],

           [[ 3211,    47,    19,     0],
            [ 3212,    46,    20,     0],
            [ 3213,    43,    21,    13],
            [ 3214,    44,     1,     0],
            [ 3215,    45,     1,     0]],

           ..., 

