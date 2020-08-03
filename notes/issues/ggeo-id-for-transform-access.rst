ggeo-id-for-transform-access
=============================

Motivation
------------

* need an identifier to facilitate accessing global transforms of any volume.
* this is to provide instance "cathode" frame local positions for hits 
  without doubling the photon 4x4 for the locals 

  * instead label geometry with a ggeo_id that allows to access the transform (and its inverse)


Why not just use nodeIdx ? (from the full traversal)
--------------------------------------------------------

* too many nodes ? 316326
* no higher meaning 
* no hope of any stability, nodeIdx will change with almost every geometry change
* comparing with transforms looked up via nodeIdx is the way to check the ggeo_id route 
* probably best to do it two ways and cross-check them anyhow

::

    epsilon:1 blyth$ inp GMergedMesh/?/iidentity.npy
    a :                                  GMergedMesh/1/iidentity.npy :        (25600, 5, 4) : a4a7deb934cae243b9181c80ddc1066b : 20200730-1543 
    b :                                  GMergedMesh/2/iidentity.npy :        (12612, 6, 4) : 4423ba6434c39aff488e6784df468ae1 : 20200730-1543 
    c :                                  GMergedMesh/3/iidentity.npy :         (5000, 6, 4) : 52c59e1bb3179c404722c2df4c26ac81 : 20200730-1543 
    d :                                  GMergedMesh/4/iidentity.npy :         (2400, 6, 4) : 08846aa446e53c50c1a7cea89674a398 : 20200730-1543 
    e :                                  GMergedMesh/5/iidentity.npy :          (590, 1, 4) : 6b57bfe28d74e9e161a1a0908d568b84 : 20200730-1543 
    f :                                  GMergedMesh/6/iidentity.npy :          (590, 1, 4) : 45836c662ac5095c0d623bf7ed8a3399 : 20200730-1543 
    g :                                  GMergedMesh/7/iidentity.npy :          (590, 1, 4) : 92bdabddd8393af96cd10f43b8e920f2 : 20200730-1543 
    h :                                  GMergedMesh/8/iidentity.npy :          (590, 1, 4) : 98a9c18bdf1d64f1fa80a10799073b8d : 20200730-1543 
    i :                                  GMergedMesh/9/iidentity.npy :        (504, 130, 4) : 01278331416251ff7fd611fd2b1debd4 : 20200730-1543 
    j :                                  GMergedMesh/0/iidentity.npy :       (1, 316326, 4) : 57ddfde998a9f5ceab681b00b3b49e5b : 20200730-1543 

    In [1]: 



Idea : compose a 32-bit field encoding 3 integers : ridx/iidx/vidx
-------------------------------------------------------------------------

repeatIdx/ridx
    repeat index (0 for global) (allow 8 bits, 256)

instanceIdx/iidx
    instance index, (allow 16 bits, 65536)

primIdx/volumeIdx/pidx
    offset volume within an instance (allow 8 bits, 256) 


::
      rr  iiii  vv
   0x(ff)(ffff)(ff)


* 8+16+8 would work for JUNO instances


Problem with this: 

* for globals, ridx=0, iidx=0 so are wasting 8+16=24 bits 
* need to adapt to use the (iiii) bits in global case



::

   0x1 << 4  =         16 
   0x1 << 8  =        256 
   0x1 << 12 =       4096
   0x1 << 16 =      65536
   0x1 << 20 =    1048576   ~1M
   0x1 << 24 =   16777216   ~16M
   0x1 << 28 =  268435456   ~268M
   0x1 << 32 = 4294967296   ~4294M  



What about globals ridx 0 ?
-----------------------------

* global arrays are special, because the higher level ones cover all volumes not just ridx 0 selected ones

* zero is too special : kinda need GMergedMesh -1 to hold what 0 currently holds and have a zero that applies the ridx selection  

  * want to be able to treat ridx zero as just another instance for which there happens to only be one of them 

* makes me wonder if the global transforms might be incorrect 

* -1 would be too problematic, use nmm+1 


::

    epsilon:1 blyth$ inp GMergedMesh/0/*.npy
    a :                                       GMergedMesh/0/bbox.npy :          (316326, 6) : e03de9c79f6f50a14d0ccbc6ed482e09 : 20200730-1543 
    b :                              GMergedMesh/0/center_extent.npy :          (316326, 4) : cf16b7b71b30d3de903b1fcac6b84db8 : 20200730-1543 
    c :                                   GMergedMesh/0/identity.npy :          (316326, 4) : 2a0515dd3da7723f1e6430ecb14536fa : 20200730-1543 
    d :                                     GMergedMesh/0/meshes.npy :          (316326, 1) : 769ac94734ee1d4df8f43922921d739c : 20200730-1543 
    e :                                   GMergedMesh/0/nodeinfo.npy :          (316326, 4) : 5f2019eddf04b4d59a28114107d3d962 : 20200730-1543 
    f :                                 GMergedMesh/0/transforms.npy :         (316326, 16) : 5b55d80e152bfc1edb08acd50423fa7b : 20200730-1543 
    g :                                    GMergedMesh/0/indices.npy :          (150408, 1) : 2af831a56809847c4bac31ed8b75391d : 20200730-1543 
    h :                                 GMergedMesh/0/boundaries.npy :           (50136, 1) : ec86774a4b541196fe19060a45f80c9f : 20200730-1543 
    i :                                      GMergedMesh/0/nodes.npy :           (50136, 1) : 20b5a07b5fd9a591316ef813f917e09f : 20200730-1543 
    j :                                    GMergedMesh/0/sensors.npy :           (50136, 1) : 5a535a6294a983f85a9d39594f5f2025 : 20200730-1543 
    k :                                     GMergedMesh/0/colors.npy :           (26138, 3) : 70b5ff210429c7018832882046c73830 : 20200730-1543 
    l :                                    GMergedMesh/0/normals.npy :           (26138, 3) : 40035f80ada1486bb9abcca02cb5890b : 20200730-1543 
    m :                                   GMergedMesh/0/vertices.npy :           (26138, 3) : 2929dbcd7b89ddd816cdf59c88e1bed6 : 20200730-1543 
    n :                                  GMergedMesh/0/iidentity.npy :       (1, 316326, 4) : 57ddfde998a9f5ceab681b00b3b49e5b : 20200730-1543 
    o :                                GMergedMesh/0/itransforms.npy :            (1, 4, 4) : 2142ffd110056f6eba647180adfbbcc9 : 20200730-1543 


    epsilon:1 blyth$ inp GMergedMesh/1/*.npy
    a :                                  GMergedMesh/1/iidentity.npy :        (25600, 5, 4) : a4a7deb934cae243b9181c80ddc1066b : 20200730-1543 
    b :                                GMergedMesh/1/itransforms.npy :        (25600, 4, 4) : 29a7bf21dabfd4a6f9228fadb7edabca : 20200730-1543 
    c :                                    GMergedMesh/1/indices.npy :            (4752, 1) : b5d5dc7ce94690319fb384b1e503e2f9 : 20200730-1543 
    d :                                 GMergedMesh/1/boundaries.npy :            (1584, 1) : 4583b9e4b2524fc02d90306a4ae93238 : 20200730-1543 
    e :                                      GMergedMesh/1/nodes.npy :            (1584, 1) : 8cb9bf708067a07977010b6bc92bf565 : 20200730-1543 
    f :                                    GMergedMesh/1/sensors.npy :            (1584, 1) : 5013e90692fee549bfd43714d7c8aa3d : 20200730-1543 
    g :                                     GMergedMesh/1/colors.npy :             (805, 3) : 5b2f1391f85c6e29560eed612a0e890a : 20200730-1543 
    h :                                    GMergedMesh/1/normals.npy :             (805, 3) : 5482a46493c73523fdc5356fd6ed5ebc : 20200730-1543 
    i :                                   GMergedMesh/1/vertices.npy :             (805, 3) : b447acf665678da2789103b44874d6bb : 20200730-1543 
    j :                                       GMergedMesh/1/bbox.npy :               (5, 6) : a523db9c1220c034d29d8c0113b4ac10 : 20200730-1543 
    k :                              GMergedMesh/1/center_extent.npy :               (5, 4) : 3417b940f4da6db67abcf29937b52128 : 20200730-1543 
    l :                                   GMergedMesh/1/identity.npy :               (5, 4) : a921a71d379336f28e7c0b908eea9218 : 20200730-1543 
    m :                                     GMergedMesh/1/meshes.npy :               (5, 1) : 0a52a5397e61677ded7cd8a7b23bf090 : 20200730-1543 
    n :                                   GMergedMesh/1/nodeinfo.npy :               (5, 4) : c143e214851e70197a6de58b2c86b5a9 : 20200730-1543 
    o :                                 GMergedMesh/1/transforms.npy :              (5, 16) : 37ae1f7f4da2409596627cebfa5cb28b : 20200730-1543 



non-zero ridx accessing a transform from identity triplet
----------------------------------------------------------

* ridx -> which GMergedMesh to access
* iidx -> itransforms.npy index and iidentity.npy index
* pidx -> transforms.npy index   
* multiply the two transforms 

::

    cd $GC

    epsilon:1 blyth$ inp GMergedMesh/1/*.npy
    a :                                  GMergedMesh/1/iidentity.npy :        (25600, 5, 4) : a4a7deb934cae243b9181c80ddc1066b : 20200730-1543 
    b :                                GMergedMesh/1/itransforms.npy :        (25600, 4, 4) : 29a7bf21dabfd4a6f9228fadb7edabca : 20200730-1543 
    c :                                    GMergedMesh/1/indices.npy :            (4752, 1) : b5d5dc7ce94690319fb384b1e503e2f9 : 20200730-1543 
    d :                                 GMergedMesh/1/boundaries.npy :            (1584, 1) : 4583b9e4b2524fc02d90306a4ae93238 : 20200730-1543 
    e :                                      GMergedMesh/1/nodes.npy :            (1584, 1) : 8cb9bf708067a07977010b6bc92bf565 : 20200730-1543 
    f :                                    GMergedMesh/1/sensors.npy :            (1584, 1) : 5013e90692fee549bfd43714d7c8aa3d : 20200730-1543 
    g :                                     GMergedMesh/1/colors.npy :             (805, 3) : 5b2f1391f85c6e29560eed612a0e890a : 20200730-1543 
    h :                                    GMergedMesh/1/normals.npy :             (805, 3) : 5482a46493c73523fdc5356fd6ed5ebc : 20200730-1543 
    i :                                   GMergedMesh/1/vertices.npy :             (805, 3) : b447acf665678da2789103b44874d6bb : 20200730-1543 
    j :                                       GMergedMesh/1/bbox.npy :               (5, 6) : a523db9c1220c034d29d8c0113b4ac10 : 20200730-1543 
    k :                              GMergedMesh/1/center_extent.npy :               (5, 4) : 3417b940f4da6db67abcf29937b52128 : 20200730-1543 
    l :                                   GMergedMesh/1/identity.npy :               (5, 4) : a921a71d379336f28e7c0b908eea9218 : 20200730-1543 
    m :                                     GMergedMesh/1/meshes.npy :               (5, 1) : 0a52a5397e61677ded7cd8a7b23bf090 : 20200730-1543 
    n :                                   GMergedMesh/1/nodeinfo.npy :               (5, 4) : c143e214851e70197a6de58b2c86b5a9 : 20200730-1543 
    o :                                 GMergedMesh/1/transforms.npy :              (5, 16) : 37ae1f7f4da2409596627cebfa5cb28b : 20200730-1543 

    In [1]: o   ## all identity within ridx 1 instance : so not good for testing 
    Out[1]: 
    array([[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
           [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
           [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
           [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
           [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]], dtype=float32)

    In [1]: a
    Out[1]: 
    array([[[173922,     40,     21, 300000],
            [173923,     38,     22, 300000],
            [173924,     36,     28, 300000],
            [173925,     37,     29, 300000],
            [173926,     39,     19, 300000]],

           [[173927,     40,     21, 300001],
            [173928,     38,     22, 300001],
            [173929,     36,     28, 300001],
            [173930,     37,     29, 300001],
            [173931,     39,     19, 300001]],

::

    In [4]: ridx,iidx,pidx = 1,100,3

    In [6]: ii = np.load(os.path.expandvars("$GC/GMergedMesh/%(ridx)s/iidentity.npy" % locals()))

    In [9]: ii[iidx,pidx]
    Out[9]: array([174425,     37,     29, 300100], dtype=uint32)

    In [10]: ii[iidx,pidx,0]
    Out[10]: 174425

    In [11]: nidx = ii[iidx,pidx,0]

    In [12]: nidx
    Out[12]: 174425

    In [14]: gt = np.load(os.path.expandvars("$GC/GMergedMesh/0/transforms.npy"))   ## absolute addressing in ridx 0 is convenient 

    In [15]: gt.shape
    Out[15]: (316326, 16)

    In [17]: gt[nidx].reshape(4,4)
    Out[17]: 
    array([[     0.9067,     -0.3632,      0.2147,      0.    ],
           [     0.3719,      0.9283,      0.    ,      0.    ],
           [    -0.1993,      0.0798,      0.9767,      0.    ],
           [  3862.4187,  -1547.188 , -18932.178 ,      1.    ]], dtype=float32)


    In [18]: it = np.load(os.path.expandvars("$GC/GMergedMesh/%(ridx)s/itransforms.npy" % locals()))

    In [19]: it.shape
    Out[19]: (25600, 4, 4)

    In [20]: it[iidx]
    Out[20]: 
    array([[     0.9067,     -0.3632,      0.2147,      0.    ],
           [     0.3719,      0.9283,      0.    ,      0.    ],
           [    -0.1993,      0.0798,      0.9767,      0.    ],
           [  3862.4187,  -1547.188 , -18932.178 ,      1.    ]], dtype=float32)

    In [21]: 

    In [21]: vt = np.load(os.path.expandvars("$GC/GMergedMesh/%(ridx)s/transforms.npy" % locals()))

    In [22]: vt   ## all vt other than ridx 0 and ridx 9 are identity : so need to check those for a proper test
    Out[22]: 
    array([[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
           [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
           [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
           [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
           [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]], dtype=float32)


ridx 9 : pick a volume and try to find its transform in two ways
--------------------------------------------------------------------

::

    epsilon:1 blyth$ inp GMergedMesh/9/*.npy
    a :                                    GMergedMesh/9/indices.npy :            (4680, 1) : 5111e266442d2c841eb83d8c2354af94 : 20200730-1543 
    b :                                 GMergedMesh/9/boundaries.npy :            (1560, 1) : b347ce265ebf37b77625d6635a6ef7a1 : 20200730-1543 
    c :                                      GMergedMesh/9/nodes.npy :            (1560, 1) : ff96913e54a942e9b032068130afd493 : 20200730-1543 
    d :                                    GMergedMesh/9/sensors.npy :            (1560, 1) : 554d2c41de0447fa9f39c2e4e703d727 : 20200730-1543 
    e :                                     GMergedMesh/9/colors.npy :            (1040, 3) : dc88b23194d56a88ea939fb0bc569960 : 20200730-1543 
    f :                                    GMergedMesh/9/normals.npy :            (1040, 3) : 8851ad10adca2946f46f5c0f0a8e2603 : 20200730-1543 
    g :                                   GMergedMesh/9/vertices.npy :            (1040, 3) : afea85cc5ee1a9cd1f5f88d471cddd9f : 20200730-1543 
    h :                                  GMergedMesh/9/iidentity.npy :        (504, 130, 4) : 01278331416251ff7fd611fd2b1debd4 : 20200730-1543 
    i :                                GMergedMesh/9/itransforms.npy :          (504, 4, 4) : f6752ff4aaef420338d38431219675aa : 20200730-1543 
    j :                                       GMergedMesh/9/bbox.npy :             (130, 6) : 7bc56c6ee33c5f67109b70ec9e185c9e : 20200730-1543 
    k :                              GMergedMesh/9/center_extent.npy :             (130, 4) : 1018dd0512c2ec73bdcab664b941ea89 : 20200730-1543 
    l :                                   GMergedMesh/9/identity.npy :             (130, 4) : 52970fab4ed00fecce40f89f36b77055 : 20200730-1543 
    m :                                     GMergedMesh/9/meshes.npy :             (130, 1) : 09ba276e804657d0d238d70e6237d64e : 20200730-1543 
    n :                                   GMergedMesh/9/nodeinfo.npy :             (130, 4) : d0ea6aff888be261bf22b8324b9926f3 : 20200730-1543 
    o :                                 GMergedMesh/9/transforms.npy :            (130, 16) : ecce39f876a3b241c76dd6f11ee214d3 : 20200730-1543 



    In [1]: ridx,iidx,pidx = 9,503,129

    In [2]: it = np.load(os.path.expandvars("$GC/GMergedMesh/%(ridx)s/itransforms.npy" % locals()))

    In [3]: ii = np.load(os.path.expandvars("$GC/GMergedMesh/%(ridx)s/iidentity.npy" % locals()))

    In [4]: vt = np.load(os.path.expandvars("$GC/GMergedMesh/%(ridx)s/transforms.npy" % locals()))

    In [5]: gt = np.load(os.path.expandvars("$GC/GMergedMesh/0/transforms.npy"))

    In [6]: it[iidx]
    Out[6]: 
    array([[     0.  ,      1.  ,      0.  ,      0.  ],
           [    -1.  ,      0.  ,      0.  ,      0.  ],
           [     0.  ,      0.  ,      1.  ,      0.  ],
           [-22672.5 ,   6711.2 ,  26504.15,      1.  ]], dtype=float32)

    In [18]: vt[pidx].reshape(4,4)              # y shift and no rotation within the instance
    Out[18]: 
    array([[  1. ,   0. ,   0. ,   0. ],
           [  0. ,   1. ,   0. ,   0. ],
           [  0. ,   0. ,   1. ,   0. ],
           [  0. , 831.6,   0. ,   1. ]], dtype=float32)

    In [10]: np.dot(  it[iidx], vt[pidx].reshape(4,4) )     ## could be the wrong way round ?
    Out[10]: 
    array([[     0.    ,      1.    ,      0.    ,      0.    ],
           [    -1.    ,      0.    ,      0.    ,      0.    ],
           [     0.    ,      0.    ,      1.    ,      0.    ],
           [-22672.5   ,   7542.8003,  26504.15  ,      1.    ]], dtype=float32)

    In [11]: 6711.2+831.6
    Out[11]: 7542.8

    In [17]: np.dot( vt[pidx].reshape(4,4), it[iidx] )    ## multiply in other order results in a shift in x rather than z because of the axis rotations 
    Out[17]: 
    array([[     0.  ,      1.  ,      0.  ,      0.  ],
           [    -1.  ,      0.  ,      0.  ,      0.  ],
           [     0.  ,      0.  ,      1.  ,      0.  ],
           [-23504.1 ,   6711.2 ,  26504.15,      1.  ]], dtype=float32)


    In [12]: nidx = ii[iidx,pidx,0]

    In [13]: nidx
    Out[13]: 65716

    In [16]: gt[nidx].reshape(4,4)             ## matches the 2nd ordering 
    Out[16]: 
    array([[     0.  ,      1.  ,      0.  ,      0.  ],
           [    -1.  ,      0.  ,      0.  ,      0.  ],
           [     0.  ,      0.  ,      1.  ,      0.  ],
           [-23504.1 ,   6711.2 ,  26504.15,      1.  ]], dtype=float32)



    In [9]: np.set_printoptions(linewidth=200, edgeitems=100 )  ## lots of different z-shifts for the 130 volumes within the 504 instances

    In [10]: o[:130]
    Out[10]: 
    array([[   1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ],
           [   1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ],
           [   1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. , -831.6,    0. ,    1. ],
           [   1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. , -831.6,    0. ,    1. ],
           [   1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. , -805.2,    0. ,    1. ],
           [   1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. , -805.2,    0. ,    1. ],
           [   1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. , -778.8,    0. ,    1. ],
           [   1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. , -778.8,    0. ,    1. ],
           [   1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. , -752.4,    0. ,    1. ],
           [   1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. ,    0. ,    0. ,    1. ,    0. ,    0. , -752.4,    0. ,    1. ],


    In [16]: np.set_printoptions(linewidth=300, edgeitems=1000)   ## bunch of axis flips/swaps and translations for the 504 instances

    In [17]: i.reshape(-1,16)
    Out[17]: 
    array([[     1.    ,      0.    ,      0.    ,      0.    ,      0.    ,      1.    ,      0.    ,      0.    ,      0.    ,      0.    ,      1.    ,      0.    ,  20133.6   ,  -9250.1   ,  23489.85  ,      1.    ],
           [     1.    ,      0.    ,      0.    ,      0.    ,      0.    ,      1.    ,      0.    ,      0.    ,      0.    ,      0.    ,      1.    ,      0.    ,  20133.6   ,  -7557.5   ,  23489.85  ,      1.    ],
           [     1.    ,      0.    ,      0.    ,      0.    ,      0.    ,      1.    ,      0.    ,      0.    ,      0.    ,      0.    ,      1.    ,      0.    ,  20133.6   ,  -5864.9004,  23489.85  ,      1.    ],
           [     1.    ,      0.    ,      0.    ,      0.    ,      0.    ,      1.    ,      0.    ,      0.    ,      0.    ,      0.    ,      1.    ,      0.    ,  20133.6   ,  -4172.3003,  23489.85  ,      1.    ],
           [     0.    ,      1.    ,      0.    ,      0.    ,     -1.    ,      0.    ,      0.    ,      0.    ,      0.    ,      0.    ,      1.    ,      0.    ,  22672.5   ,  -6711.2   ,  23504.15  ,      1.    ],
           [     0.    ,      1.    ,      0.    ,      0.    ,     -1.    ,      0.    ,      0.    ,      0.    ,      0.    ,      0.    ,      1.    ,      0.    ,  20979.9   ,  -6711.2   ,  23504.15  ,      1.    ],
           [     0.    ,      1.    ,      0.    ,      0.    ,     -1.    ,      0.    ,      0.    ,      0.    ,      0.    ,      0.    ,      1.    ,      0.    ,  19287.299 ,  -6711.2   ,  23504.15  ,      1.    ],
           [     0.    ,      1.    ,      0.    ,      0.    ,     -1.    ,      0.    ,      0.    ,      0.    ,      0.    ,      0.    ,      1.    ,      0.    ,  17594.7   ,  -6711.2   ,  23504.15  ,      1.    ],
           [     1.    ,      0.    ,      0.    ,      0.    ,      0.    ,      1.    ,      0.    ,      0.    ,      0.    ,      0.    ,      1.    ,      0.    ,  13422.4   ,  -9250.1   ,  23439.85  ,      1.    ],
           [     1.    ,      0.    ,      0.    ,      0.    ,      0.    ,      1.    ,      0.    ,      0.    ,      0.    ,      0.    ,      1.    ,      0.    ,  13422.4   ,  -7557.5   ,  23439.85  ,      1.    ],
           [     1.    ,      0.    ,      0.    ,      0.    ,      0.    ,      1.    ,      0.    ,      0.    ,      0.    ,      0.    ,      1.    ,      0.    ,  13422.4   ,  -5864.9004,  23439.85  ,      1.    ],
           [     1.    ,      0.    ,      0.    ,      0.    ,      0.    ,      1.    ,      0.    ,      0.    ,      0.    ,      0.    ,      1.    ,      0.    ,  13422.4   ,  -4172.3003,  23439.85  ,      1.    ],



ridx 0 : pick a volume : nope
-------------------------------

* not able to do this for ridx 0, as ridx 0 arrays contain everything not just those with ridx=0 
* such global arrays only give access by nidx
* it is not even possible to see which volumes are globals and which are instanced

::

    epsilon:1 blyth$ inp GMergedMesh/0/*.npy
    a :                                       GMergedMesh/0/bbox.npy :          (316326, 6) : cc56f52ec9eaaf3cd308e74fbdeb7afb : 20200719-2129 
    b :                              GMergedMesh/0/center_extent.npy :          (316326, 4) : ed0e99d0e81782a5e4081511aef91b9e : 20200719-2129 
    c :                                   GMergedMesh/0/identity.npy :          (316326, 4) : 2a0515dd3da7723f1e6430ecb14536fa : 20200719-2129 
    d :                                     GMergedMesh/0/meshes.npy :          (316326, 1) : 769ac94734ee1d4df8f43922921d739c : 20200719-2129 
    e :                                   GMergedMesh/0/nodeinfo.npy :          (316326, 4) : 5f2019eddf04b4d59a28114107d3d962 : 20200719-2129 
    f :                                 GMergedMesh/0/transforms.npy :         (316326, 16) : 5b55d80e152bfc1edb08acd50423fa7b : 20200719-2129 
    g :                                    GMergedMesh/0/indices.npy :          (150408, 1) : 2af831a56809847c4bac31ed8b75391d : 20200719-2129 
    h :                                 GMergedMesh/0/boundaries.npy :           (50136, 1) : ec86774a4b541196fe19060a45f80c9f : 20200719-2129 
    i :                                      GMergedMesh/0/nodes.npy :           (50136, 1) : 20b5a07b5fd9a591316ef813f917e09f : 20200719-2129 
    j :                                    GMergedMesh/0/sensors.npy :           (50136, 1) : 92b01be7ab2c281a45c51434f293049f : 20200719-2129 
    k :                                     GMergedMesh/0/colors.npy :           (26138, 3) : 70b5ff210429c7018832882046c73830 : 20200719-2129 
    l :                                    GMergedMesh/0/normals.npy :           (26138, 3) : 40035f80ada1486bb9abcca02cb5890b : 20200719-2129 
    m :                                   GMergedMesh/0/vertices.npy :           (26138, 3) : 2929dbcd7b89ddd816cdf59c88e1bed6 : 20200719-2129 
    n :                                  GMergedMesh/0/iidentity.npy :       (1, 316326, 4) : 57ddfde998a9f5ceab681b00b3b49e5b : 20200719-2129 
    o :                                GMergedMesh/0/itransforms.npy :            (1, 4, 4) : 2142ffd110056f6eba647180adfbbcc9 : 20200719-2129 


    In [3]: n[0]     ## identiy quad of all volumes
    Out[3]: 
    array([[     0,     56,      0,      0],
           [     1,     12,      1,      0],
           [     2,     11,      2,      0],
           ...,
           [316323,     50,     23,  32399],
           [316324,     48,     33,  32399],
           [316325,     49,     34,  32399]], dtype=uint32)

    In [4]: n[0].shape
    Out[4]: (316326, 4)


    In [1]: ridx,iidx,pidx = 0,0,129

    In [2]: it = np.load(os.path.expandvars("$GC/GMergedMesh/%(ridx)s/itransforms.npy" % locals()))   # identity matrix

    In [3]: ii = np.load(os.path.expandvars("$GC/GMergedMesh/%(ridx)s/iidentity.npy" % locals()))

    In [4]: vt = np.load(os.path.expandvars("$GC/GMergedMesh/%(ridx)s/transforms.npy" % locals()))

    In [5]: gt = np.load(os.path.expandvars("$GC/GMergedMesh/0/transforms.npy"))



Need to make ridx 0 less special, but still keep the globals 
----------------------------------------------------------------

Can the global pathways be brought in line with the instanced ?::

     713 void Scene::uploadGeometry()
     714 {
     715     // invoked by OpticksViz::uploadGeometry
     716     assert(m_geolib && "must setGeometry first");
     717     unsigned int nmm = m_geolib->getNumMergedMesh();
     718 
     719     LOG(info) << " nmm " << nmm ;
     720 
     721     //m_geolib->dump("Scene::uploadGeometry GGeoLib" );
     722 
     723     m_context->init();  // UBO setup
     724 
     725 
     726     for(unsigned int i=0 ; i < nmm ; i++)
     727     {
     728         GMergedMesh* mm = m_geolib->getMergedMesh(i);
     729         if(!mm) continue ;
     730 
     731         LOG(debug) << i << " geoCode " << mm->getGeoCode() ;
     732 
     733         if( i == 0 )  // first mesh assumed to be **the one and only** non-instanced global mesh
     734         {
     735            uploadGeometryGlobal(mm);
     736         }
     737         else
     738         {
     739            uploadGeometryInstanced(mm);
     740         }
     741     }
     742 



How does Global rendering differ from Instanced ? Especially wrt the input buffer data.
-------------------------------------------------------------------------------------------

* global uses m_global_renderer(nrm) + m_globalvec_renderer(nrmvec) 
* instanced uses m_instance_renderer(inrm) + m_bbox_renderer(inrm)


::

     322 void Renderer::setDrawable(GDrawable* drawable)
     323 {  
     324     assert(drawable); 
     325     m_drawable = drawable ;
     326    
     327     NSlice* islice = drawable->getInstanceSlice();
     328     NSlice* fslice = drawable->getFaceSlice();
     329    
     330     //  nvert: vertices, normals, colors
     331     m_vbuf = MAKE_RBUF(m_drawable->getVerticesBuffer());
     332     m_nbuf = MAKE_RBUF(m_drawable->getNormalsBuffer());
     333     m_cbuf = MAKE_RBUF(m_drawable->getColorsBuffer());
     334    
     335     assert(m_vbuf->getNumBytes() == m_cbuf->getNumBytes());
     336     assert(m_nbuf->getNumBytes() == m_cbuf->getNumBytes());
     337    
     338     // 3*nface indices
     339     GBuffer* fbuf_orig = m_drawable->getIndicesBuffer();
     340     GBuffer* fbuf = fslice ? fslice_element_buffer(fbuf_orig, fslice) : fbuf_orig ;
     341 
     342     m_fbuf = MAKE_RBUF(fbuf) ;
     343    
     344     m_tbuf = MAKE_RBUF(m_drawable->getTexcoordsBuffer());
     345     setHasTex(m_tbuf != NULL);
     346 
     347     NPY<float>* ibuf_orig = m_drawable->getITransformsBuffer();
     348 
     349     if(islice)
     350         LOG(warning) << "Renderer::setDrawable instance slicing ibuf with " << islice->description() ;
     351 
     352     NPY<float>* ibuf = islice ? ibuf_orig->make_slice(islice) :  ibuf_orig ;
     353     if(ibuf) ibuf->setName("itransforms");
     354 
     355     m_ibuf = MAKE_RBUF(ibuf) ;
     356     setHasTransforms(m_ibuf != NULL);
     357 }



GInstancer
------------

::


 333    m_desc.add_options()
 334        ("globalinstance",
 335         "Plus one GMergedMesh collected from nodes labelled with ridx 0, ie a non-special global treated as instanced,"
 336         "but with only one identity instance transform ") ;
 337         




First try at globainstance
----------------------------



::

    geocache-tds --globalinstance

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff532bbb66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff53486080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff532171ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff531df1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001005437f1 libOptiXRap.dylib`OGeo::makeAnalyticGeometry(this=0x000000028da36e30, mm=0x000000020d4c1720, lod=0) at OGeo.cc:715
        frame #5: 0x0000000100540406 libOptiXRap.dylib`OGeo::makeOGeometry(this=0x000000028da36e30, mergedmesh=0x000000020d4c1720, lod=0) at OGeo.cc:599
        frame #6: 0x000000010053f2b9 libOptiXRap.dylib`OGeo::makeRepeatedAssembly(this=0x000000028da36e30, mm=0x000000020d4c1720, raylod=false) at OGeo.cc:377
        frame #7: 0x000000010053db01 libOptiXRap.dylib`OGeo::convertMergedMesh(this=0x000000028da36e30, i=10) at OGeo.cc:309
        frame #8: 0x000000010053d39d libOptiXRap.dylib`OGeo::convert(this=0x000000028da36e30) at OGeo.cc:270
        frame #9: 0x0000000100533269 libOptiXRap.dylib`OScene::init(this=0x000000028d8dfd50) at OScene.cc:169
        frame #10: 0x0000000100532621 libOptiXRap.dylib`OScene::OScene(this=0x000000028d8dfd50, hub=0x000000028d31cfb0, cmake_target="OptiXRap", ptxrel=0x0000000000000000) at OScene.cc:91
        frame #11: 0x000000010053383d libOptiXRap.dylib`OScene::OScene(this=0x000000028d8dfd50, hub=0x000000028d31cfb0, cmake_target="OptiXRap", ptxrel=0x0000000000000000) at OScene.cc:90
        frame #12: 0x0000000100445c16 libOKOP.dylib`OpEngine::OpEngine(this=0x000000028d8df200, hub=0x000000028d31cfb0) at OpEngine.cc:75
        frame #13: 0x000000010044630d libOKOP.dylib`OpEngine::OpEngine(this=0x000000028d8df200, hub=0x000000028d31cfb0) at OpEngine.cc:83
        frame #14: 0x000000010010cfaf libOK.dylib`OKPropagator::OKPropagator(this=0x000000028d8df920, hub=0x000000028d31cfb0, idx=0x000000028d83d450, viz=0x000000028d83d470) at OKPropagator.cc:68
        frame #15: 0x000000010010d15d libOK.dylib`OKPropagator::OKPropagator(this=0x000000028d8df920, hub=0x000000028d31cfb0, idx=0x000000028d83d450, viz=0x000000028d83d470) at OKPropagator.cc:72
        frame #16: 0x000000010010c06c libOK.dylib`OKMgr::OKMgr(this=0x00007ffeefbfe0e0, argc=8, argv=0x00007ffeefbfe930, argforced=0x0000000000000000) at OKMgr.cc:63
        frame #17: 0x000000010010c4db libOK.dylib`OKMgr::OKMgr(this=0x00007ffeefbfe0e0, argc=8, argv=0x00007ffeefbfe930, argforced=0x0000000000000000) at OKMgr.cc:65
        frame #18: 0x0000000100015a50 OKX4Test`main(argc=8, argv=0x00007ffeefbfe930) at OKX4Test.cc:116
        frame #19: 0x00007fff5316b015 libdyld.dylib`start + 1
    (lldb) 


