gxr_shakedown
================


::

    gx
    ./gxr.sh dbg




FIXED : issue 1 : output path unsustainable : Changed to use DefaultOutputDir
-----------------------------------------------------------------------------------

::

    2022-07-09 22:13:47.452 INFO  [375788] [CSGOptiX::launch@718]  (width, height, depth) ( 1920,1080,1) 0.0078
    2022-07-09 22:13:47.452 ERROR [375788] [CSGOptiX::render_snap@797]  name cx-1 outpath /tmp/blyth/opticks/cx-1.jpg dt 0.00778148 topline [G4CXRenderTest] botline [    0.0078]
    2022-07-09 22:13:47.452 INFO  [375788] [CSGOptiX::snap@819]  path /tmp/blyth/opticks/cx-1.jpg
    2022-07-09 22:13:47.452 INFO  [375788] [CSGOptiX::snap@828]  path_ [/tmp/blyth/opticks/cx-1.jpg]
    2022-07-09 22:13:47.452 INFO  [375788] [CSGOptiX::snap@829]  topline G4CXRenderTestPIP  td:1 pv:2 av:2 WITH_PRD  
    NP::Write dtype <f4 ni     1080 nj  1920 nk  4 nl  -1 nm  -1 no  -1 path /tmp/blyth/opticks/isect.npy
    NP::Write dtype <f4 ni     1080 nj  1920 nk  4 nl  4 nm  -1 no  -1 path /tmp/blyth/opticks/photon.npy
    2022-07-09 22:13:52.592 INFO  [375788] [CSGOptiX::saveMeta@895] /tmp/blyth/opticks/cx-1.json
    N[blyth@localhost g4cx]$ 


FIXED : issue 2 : frame photons not being populated but still being written : WITH_FRAME_PHOTON
--------------------------------------------------------------------------------------------------


NOT AN ISSUE : issue 3 : isect prd.identity looks unreasonable having only 2 unique values : seems discrepant with the render
------------------------------------------------------------------------------------------------------------------------------

Actually looking again having two values is expected, the rock box surround and the PMT with primIdx values of 1 and 2 

::

    In [7]: t.isect.view(np.uint32)[:,:,3] >> 16 
    Out[7]: 
    array([[1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1],
           ...,
           [1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1]], dtype=uint32)

    In [8]: np.unique( t.isect.view(np.uint32)[:,:,3] >> 16 , return_counts=True )
    Out[8]: (array([1, 2], dtype=uint32), array([1993260,   80340]))


::

    In [1]: t.isect[0]
    Out[1]: 
    array([[ -237.5  ,  1000.   ,   562.5  ,     0.   ],
           [ -236.913,  1000.   ,   563.087,     0.   ],
           [ -236.326,  1000.   ,   563.674,     0.   ],
           [ -235.737,  1000.   ,   564.263,     0.   ],
           [ -235.146,  1000.   ,   564.854,     0.   ],
           ...,
           [ -234.555, -1000.   ,   565.445,     0.   ],
           [ -235.147, -1000.   ,   564.853,     0.   ],
           [ -235.737, -1000.   ,   564.263,     0.   ],
           [ -236.326, -1000.   ,   563.674,     0.   ],

    In [2]: t.isect.view(np.int32)[:,:,3]
    Out[2]: 
    array([[65536, 65536, 65536, 65536, 65536, ..., 65536, 65536, 65536, 65536, 65536],
           [65536, 65536, 65536, 65536, 65536, ..., 65536, 65536, 65536, 65536, 65536],
           [65536, 65536, 65536, 65536, 65536, ..., 65536, 65536, 65536, 65536, 65536],
           [65536, 65536, 65536, 65536, 65536, ..., 65536, 65536, 65536, 65536, 65536],
           [65536, 65536, 65536, 65536, 65536, ..., 65536, 65536, 65536, 65536, 65536],
           ...,
           [65536, 65536, 65536, 65536, 65536, ..., 65536, 65536, 65536, 65536, 65536],
           [65536, 65536, 65536, 65536, 65536, ..., 65536, 65536, 65536, 65536, 65536],
           [65536, 65536, 65536, 65536, 65536, ..., 65536, 65536, 65536, 65536, 65536],
           [65536, 65536, 65536, 65536, 65536, ..., 65536, 65536, 65536, 65536, 65536],
           [65536, 65536, 65536, 65536, 65536, ..., 65536, 65536, 65536, 65536, 65536]], dtype=int32)

    In [3]: np.unique( t.isect.view(np.int32)[:,:,3], return_counts=True )
    Out[3]: (array([ 65536, 131072], dtype=int32), array([1993260,   80340]))


