g4ok_hit_matching
===================

Context
---------

* continuation from :doc:`g4ok_investigate_zero_hits`


Recall that ckm- CerenkovMinimal is not a full debugging environment
----------------------------------------------------------------------

This was the reason for the bi-executable approach : the 2nd one being 
fully instrumented whilst leaving the first close to ordinary usage very similar
to standard Geant4 examples.

Nevertheless have a look first in minimal environment to get reminded of what is there.



The 42 and 108 ht are apparent::

    ckm-kcd

    epsilon:1 blyth$ np.py source/evt/g4live/natural/1/
    /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1/source/evt/g4live/natural/1
    source/evt/g4live/natural/1/report.txt : 38 
    source/evt/g4live/natural/1/ps.npy :          (221, 1, 4) : ad7498e182d8df1bf720c0ba0e72736c 
    source/evt/g4live/natural/1/ht.npy :           (42, 4, 4) : c8b6173ecaa8a53d8c5ea17b347fffef 
    source/evt/g4live/natural/1/rx.npy :      (221, 10, 2, 4) : c085570c57f4749d13475312fcd16fb5 
    source/evt/g4live/natural/1/fdom.npy :            (3, 1, 4) : f8c7c318e47b0ccb2c29567e87d95e67 
    source/evt/g4live/natural/1/ox.npy :          (221, 4, 4) : e4ffe716b9116955bfde6013b95ee6f7 
    source/evt/g4live/natural/1/gs.npy :            (1, 6, 4) : f21b4f0138c122a64319243596bb2228 
    source/evt/g4live/natural/1/rs.npy :      (221, 10, 1, 4) : ce59ba752de205fb16062260c6383503 
    source/evt/g4live/natural/1/ph.npy :          (221, 1, 2) : c60500d6d56b530b1c55bf6b14c34a15 
    source/evt/g4live/natural/1/idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c 
    source/evt/g4live/natural/1/20190313_204855/report.txt : 38 
    source/evt/g4live/natural/1/20190313_192952/report.txt : 38 
    source/evt/g4live/natural/1/20190313_160621/report.txt : 38 
    source/evt/g4live/natural/1/20190313_202657/report.txt : 38 
    source/evt/g4live/natural/1/20190313_144506/report.txt : 38 
    source/evt/g4live/natural/1/20190313_174223/report.txt : 38 
    source/evt/g4live/natural/1/20180928_114852/report.txt : 38 
    source/evt/g4live/natural/1/20190313_192631/report.txt : 38 
    epsilon:1 blyth$ 

    epsilon:1 blyth$ np.py source/evt/g4live/natural/-1/
    /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1/source/evt/g4live/natural/-1
    source/evt/g4live/natural/-1/ht.npy :          (108, 4, 4) : f151301a12d1874e9447fd916e7f8719 
    source/evt/g4live/natural/-1/so.npy :          (221, 4, 4) : d1b4242225f7ffc7f0ad38a9669562a4 
    epsilon:1 blyth$ 



::

    In [13]: a[:,0]
    Out[13]: 
    array([[ 303.0482,  116.9195,   90.    ,    1.53  ],
           [ 153.1065,   16.8344,   90.    ,    0.8098],
           [ 128.8066,  -28.0179,   90.    ,    0.7245],
           [ 132.6022,  -13.899 ,   90.    ,    0.7302],
           [ 248.9373,   86.2949,   90.    ,    1.2638],
           [ 210.6323,   62.5527,   90.    ,    1.0778],
           [ 129.7702,  -21.9672,   90.    ,    0.7237],
           ...
           [ 127.8584,  -37.3653,   90.    ,    0.7297],
           [ 135.6495,   -6.84  ,   90.    ,    0.7396],
           [ 128.1061,  -33.5526,   90.    ,    0.7268],
           [ 127.9607,  -43.2016,   90.    ,    0.7367],
           [ 328.9369, -337.5704,   90.    ,    2.1781],
           [ 131.8775,  -65.5113,   90.    ,    0.7834],
           [ 188.6135, -165.1583,   90.    ,    1.2091]], dtype=float32)

    In [14]: b[:,0][b[:,0,3] < 3].shape
    Out[14]: (82, 4)

    In [15]: b[:,0][b[:,0,3] < 3]
    Out[15]: 
    array([[ 201.5414,   56.3636,   90.    ,    1.0341],
           [ 188.6135, -165.1583,   90.    ,    1.2091],
           [ 131.8775,  -65.5113,   90.    ,    0.7834],
           [ 328.937 , -337.5705,   90.    ,    2.1781],
           [ 357.2595, -366.6585,  110.    ,    2.4031],
           [ 127.9607,  -43.2016,   90.    ,    0.7367],
           [ 216.3212, -201.5619,   90.    ,    1.4029],
           ...
           [ 162.852 ,   26.6267,   90.    ,    0.8532],
           [ 165.0396, -131.1898,   90.    ,    1.0405],
           [ 218.2071, -204.0852,   90.    ,    1.4164],
           [ 225.1802,   71.9594,   90.    ,    1.1482],
           [ 256.3768,   81.9377,  110.    ,    1.3389],
           [ 129.7702,  -21.9672,   90.    ,    0.7237],
           [ 210.6323,   62.5527,   90.    ,    1.0778],
           [ 134.7122,  -74.0892,   90.    ,    0.8087],
           [ 248.9373,   86.2949,   90.    ,    1.2638],
           [ 132.6022,  -13.899 ,   90.    ,    0.7302],
           [ 128.8066,  -28.0179,   90.    ,    0.7245],
           [ 153.1065,   16.8344,   90.    ,    0.8098],
           [ 255.0051, -249.729 ,   90.    ,    1.6708],
           [ 282.2347, -276.4113,  110.    ,    1.8848],
           [ 303.0482,  116.9195,   90.    ,    1.53  ],
           [ 127.6607,  -35.9956,   90.    ,    0.7276]], dtype=float32)

    In [16]: 




ckm-so : Compare the generated photons first
------------------------------------------------

::

    In [5]: a[:,:3]-b[:,:3]
    Out[5]: 
    array([[[ 0.,  0.,  0.,  0.],
            [ 0., -0.,  0.,  0.],
            [-0., -0.,  0.,  0.]],

           [[ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [-0.,  0.,  0.,  0.]],

           [[ 0.,  0.,  0.,  0.],
            [ 0., -0.,  0.,  0.],
            [ 0.,  0., -0.,  0.]],

           ...,

           [[ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [-0.,  0.,  0.,  0.]],

           [[ 0.,  0.,  0.,  0.],
            [-0., -0.,  0.,  0.],
            [-0., -0.,  0.,  0.]],

           [[ 0.,  0.,  0.,  0.],
            [ 0., -0.,  0.,  0.],
            [ 0., -0., -0.,  0.]]], dtype=float32)

    In [6]: ab = a[:,:3]-b[:,:3]

    In [7]: ab.max()
    Out[7]: 6.1035156e-05


For step by step debugging need the instrumented executable to work from gensteps.


