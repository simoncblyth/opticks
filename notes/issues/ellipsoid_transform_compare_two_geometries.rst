ellipsoid_transform_compare_two_geometries
=============================================

* from :doc:`review_geometry_translation`

Can observe the lack of ellipsoid scale transforms in a, but it does not explain why. 

BUT : notice that the A was created by a tds run from live Geant4 but B was from GDML

* TODO: redo tds running with current Opticks, getting the CF transforms correct 
  depends on --gparts-transform-offset but was previously problematic with the old tds run 
  probably : as it was half way between old model and new 

Could the explanation be "--gparts-transform-offset" again  ?

* for the old Opticks workflow : cannot use that option
* for the new Opticks workflow : that option is needed  

HMM: while the above is true, I am not very convinced that use of the  
wrong option explains the lack of ellipsoid scale transforms : 
as using the wrong option would mess up all transforms. 

HMM: but looking at the integration see that the option is not used... so 
that means that CFBase must have been created by a separate run 

TODO: minimally get tds running to use new workflow G4CXOpticks::setGeometry with GDML + CF saving  
BUT : need to integrate gx anyhow 


::

    N[blyth@localhost CSGFoundry]$ l
    total 5756
       4 drwxr-xr-x. 13 blyth blyth    4096 Jul 17 18:23 ..
    3032 -rw-rw-r--.  1 blyth blyth 3102656 Jun  2 01:50 inst.npy
     512 -rw-rw-r--.  1 blyth blyth  521856 Jun  2 01:50 itra.npy
     512 -rw-rw-r--.  1 blyth blyth  521856 Jun  2 01:50 tran.npy
    1472 -rw-rw-r--.  1 blyth blyth 1504320 Jun  2 01:50 node.npy
     204 -rw-rw-r--.  1 blyth blyth  207808 Jun  2 01:50 prim.npy
       8 -rw-rw-r--.  1 blyth blyth    4949 Jun  2 01:50 meshname.txt
       4 -rw-rw-r--.  1 blyth blyth     131 Jun  2 01:50 meta.txt
       4 -rw-rw-r--.  1 blyth blyth     190 Jun  2 01:50 mmlabel.txt
       4 -rw-rw-r--.  1 blyth blyth     608 Jun  2 01:50 solid.npy
       0 drwxr-xr-x.  2 blyth blyth     183 May 20 23:44 SSim
       0 drwxr-xr-x.  3 blyth blyth     170 May 20 23:44 .

    N[blyth@localhost CSGFoundry]$ cat meta.txt
    cxskiplv:NNVTMCPPMTsMask_virtual0x,HamamatsuR12860sMask_virtual0x,mask_PMT_20inch_vetosMask_virtual0x
    cxskiplv_idxlist:117,110,134
    N[blyth@localhost CSGFoundry]$ 


DONE: added more metadata in CSGFoundry::setMeta regarding CF creation, eg the commandline or a least executable name 



::
       
    epsilon:tests blyth$ cd ; ~/opticks/CSG/tests/CSGFoundryAB.sh 
    CSGFoundry a cfbase /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo 
    CSGFoundry b cfbase /tmp/blyth/opticks/J000/G4CXSimtraceTest 
    In [1]:                       

    In [12]: a 
    Out[12]: 
    /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/CSGFoundry
    min_stamp:2022-06-01 18:56:03.714164
    max_stamp:2022-06-01 18:56:04.071260
    age_stamp:51 days, 21:05:31.979259
             node :        (23503, 4, 4)  : /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/CSGFoundry/node.npy 
             itra :         (8152, 4, 4)  : /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/CSGFoundry/itra.npy 
         meshname :               (141,)  : /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/CSGFoundry/meshname.txt 
             meta :                 (2,)  : /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/CSGFoundry/meta.txt 
          mmlabel :                (10,)  : /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/CSGFoundry/mmlabel.txt 
             tran :         (8152, 4, 4)  : /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/CSGFoundry/tran.npy 
             inst :        (48477, 4, 4)  : /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/CSGFoundry/inst.npy 
            solid :           (10, 3, 4)  : /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/CSGFoundry/solid.npy 
             prim :         (3245, 4, 4)  : /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1/CSG_GGeo/CSGFoundry/prim.npy 

    In [13]: b
    Out[13]: 
    /tmp/blyth/opticks/J000/G4CXSimtraceTest/CSGFoundry
    min_stamp:2022-07-23 14:20:36.249319
    max_stamp:2022-07-23 14:20:41.918967
    age_stamp:1:40:54.154352
             node :        (23503, 4, 4)  : /tmp/blyth/opticks/J000/G4CXSimtraceTest/CSGFoundry/node.npy 
             itra :         (8152, 4, 4)  : /tmp/blyth/opticks/J000/G4CXSimtraceTest/CSGFoundry/itra.npy 
         meshname :               (141,)  : /tmp/blyth/opticks/J000/G4CXSimtraceTest/CSGFoundry/meshname.txt 
             meta :                 (2,)  : /tmp/blyth/opticks/J000/G4CXSimtraceTest/CSGFoundry/meta.txt 
          mmlabel :                (10,)  : /tmp/blyth/opticks/J000/G4CXSimtraceTest/CSGFoundry/mmlabel.txt 
             tran :         (8152, 4, 4)  : /tmp/blyth/opticks/J000/G4CXSimtraceTest/CSGFoundry/tran.npy 
             inst :        (48477, 4, 4)  : /tmp/blyth/opticks/J000/G4CXSimtraceTest/CSGFoundry/inst.npy 
            solid :           (10, 3, 4)  : /tmp/blyth/opticks/J000/G4CXSimtraceTest/CSGFoundry/solid.npy 
             prim :         (3245, 4, 4)  : /tmp/blyth/opticks/J000/G4CXSimtraceTest/CSGFoundry/prim.npy 
         primname :              (3245,)  : /tmp/blyth/opticks/J000/G4CXSimtraceTest/CSGFoundry/primname.txt    ## a recent addition, for quasi-B-side mocking of A-side

    In [3]: np.where( a.inst != b.inst )                                                                                                                                            
    Out[3]: (array([], dtype=int64),)

    In [22]: np.where( a.meshname != b.meshname )                                                                                                                                   
    Out[22]: (array([], dtype=int64),)

    In [23]: np.where( a.solid != b.solid )                                                                                                                                         
    Out[23]: (array([], dtype=int64), array([], dtype=int64), array([], dtype=int64))

    In [24]: np.where( a.prim != b.prim )   ## prim with ellipsoid show different bbox                                                                                                                                           
    Out[24]: 
    (array([3090, 3090, 3090, 3090, 3091, 3091, 3091, 3091, 3092, 3092, 3092, 3092, 3096, 3096, 3096, 3096, 3097, 3097, 3097, 3097, 3098, 3098, 3098, 3098, 3099, 3099, 3099, 3099, 3104, 3104, 3104, 3104]),
     array([2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3]),
     array([0, 1, 3, 0, 0, 1, 3, 0, 0, 1, 3, 0, 0, 1, 3, 0, 0, 1, 3, 0, 0, 1, 3, 0, 0, 1, 3, 0, 0, 1, 3, 0]))


    In [25]: a.prim[3090]                                                                                                                                                           
    Out[25]: 
    array([[  0.   ,   0.   ,   0.   ,   0.   ],
           [  0.   ,   0.   ,   0.   ,   0.   ],
           [-24.   , -24.   , -15.875,  24.   ],
           [ 24.   ,  24.   ,   0.   ,   0.   ]], dtype=float32)

    In [26]: b.prim[3090]                                                                                                                                                           
    Out[26]: 
    array([[  0.   ,   0.   ,   0.   ,   0.   ],
           [  0.   ,   0.   ,   0.   ,   0.   ],
           [-40.   , -40.   , -15.875,  40.   ],
           [ 40.   ,  24.   ,   0.   ,   0.   ]], dtype=float32)

    In [1]: u = np.unique(np.where(a.prim != b.prim)[0])                                                                                                                 

    In [2]: u                                                                                                                                                            
    Out[2]: array([3090, 3091, 3092, 3096, 3097, 3098, 3099, 3104])

    In [3]: a.prim[u,0,0].view(np.uint32)                                                                                                                                
    Out[3]: array([1, 1, 1, 1, 1, 1, 1, 1], dtype=uint32)

    In [4]: a.prim[u,0].view(np.uint32)                                                                                                                                  
    Out[4]: 
    array([[    1, 23210,  7950,     0],         ## numNode, nodeOffset, tranOffset, planOffset
           [    1, 23211,  7951,     0],
           [    1, 23212,  7952,     0],
           [    1, 23236,  7964,     0],
           [    1, 23237,  7965,     0],
           [    1, 23238,  7966,     0],
           [    1, 23239,  7967,     0],
           [    1, 23292,  7986,     0]], dtype=uint32)

    In [5]: b.prim[u,0].view(np.uint32)                                                                                                                                  
    Out[5]: 
    array([[    1, 23210,  7950,     0],
           [    1, 23211,  7951,     0],
           [    1, 23212,  7952,     0],
           [    1, 23236,  7964,     0],
           [    1, 23237,  7965,     0],
           [    1, 23238,  7966,     0],
           [    1, 23239,  7967,     0],
           [    1, 23292,  7986,     0]], dtype=uint32)

    ## the transforms gleaned from the tranOffset above are just with ones with scale difference


    In [6]: a.prim[u,1].view(np.uint32)                                                                                                                                  
    Out[6]: 
    array([[  1, 120,   1,   1],        ## sbtIndexOffset, meshIdx, repeatIdx, primIdx
           [  2, 118,   1,   2],
           [  3, 119,   1,   3],
           [  2, 116,   2,   3],
           [  3, 115,   2,   4],
           [  4, 113,   2,   5],
           [  5, 114,   2,   6],
           [  4, 106,   3,   5]], dtype=uint32)

    In [7]: b.prim[u,1].view(np.uint32)                                                                                                                                  
    Out[7]: 
    array([[  1, 120,   1,   1],
           [  2, 118,   1,   2],
           [  3, 119,   1,   3],
           [  2, 116,   2,   3],
           [  3, 115,   2,   4],
           [  4, 113,   2,   5],
           [  5, 114,   2,   6],
           [  4, 106,   3,   5]], dtype=uint32)





::

    In [40]: print("\n".join(b.primname[np.unique(np.where( a.prim != b.prim )[0])]))                                                                                               

    PMT_3inch_body_solid_ell_ell_helper0x66e5430
    PMT_3inch_inner1_solid_ell_helper0x66e54d0
    PMT_3inch_inner2_solid_ell_helper0x66e5570

    NNVTMCPPMT_PMT_20inch_pmt_solid_head0x5f58840
    NNVTMCPPMT_PMT_20inch_body_solid_head0x5f5a9d0
    NNVTMCPPMT_PMT_20inch_inner1_solid_head0x5f56e60
    NNVTMCPPMT_PMT_20inch_inner2_solid_head0x5f5c800

    HamamatsuR12860_PMT_20inch_inner1_solid_I0x5f39240


Bounding box difference::

    In [42]: a.prim[3104]                                                                                                                                                           
    Out[42]: 
    array([[   0.,    0.,    0.,    0.],
           [   0.,    0.,    0.,    0.],
           [-185., -185.,    0.,  185.],
           [ 185.,  185.,    0.,    0.]], dtype=float32)

    In [43]: b.prim[3104]                                                                                                                                                           
    Out[43]: 
    array([[   0.,    0.,    0.,    0.],
           [   0.,    0.,    0.,    0.],
           [-249., -249.,    0.,  249.],
           [ 249.,  185.,    0.,    0.]], dtype=float32)



    In [10]: np.where( a.itra != b.itra )                                                                                                                                           
    Out[10]: 
    (array([7950, 7950, 7951, 7951, 7952, 7952, 7964, 7964, 7965, 7965, 7966, 7966, 7967, 7967, 7986, 7986]),
     array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
     array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))

    In [11]: np.where( a.tran != b.tran )                                                                                                                                           
    Out[11]: 
    (array([7950, 7950, 7951, 7951, 7952, 7952, 7964, 7964, 7965, 7965, 7966, 7966, 7967, 7967, 7986, 7986]),
     array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
     array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))

All the transform differences are x,y scale transforms, the only scale transforms in use are for ellipsoid::

    In [1]: a.tran[7950]                                                                                                                                                            
    Out[1]: 
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]], dtype=float32)

    In [2]: b.tran[7950]                                                                                                                                                            
    Out[2]: 
    array([[1.667, 0.   , 0.   , 0.   ],
           [0.   , 1.667, 0.   , 0.   ],
           [0.   , 0.   , 1.   , 0.   ],
           [0.   , 0.   , 0.   , 1.   ]], dtype=float32)

    In [3]: a.tran[7951]                                                                                                                                                            
    Out[3]: 
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]], dtype=float32)

    In [4]: b.tran[7951]                                                                                                                                                            
    Out[4]: 
    array([[1.727, 0.   , 0.   , 0.   ],
           [0.   , 1.727, 0.   , 0.   ],
           [0.   , 0.   , 1.   , 0.   ],
           [0.   , 0.   , 0.   , 1.   ]], dtype=float32)

    In [5]: a.tran[7952]                                                                                                                                                            
    Out[5]: 
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]], dtype=float32)

    In [6]: b.tran[7952]                                                                                                                                                            
    Out[6]: 
    array([[1.727, 0.   , 0.   , 0.   ],
           [0.   , 1.727, 0.   , 0.   ],
           [0.   , 0.   , 1.   , 0.   ],
           [0.   , 0.   , 0.   , 1.   ]], dtype=float32)

    In [7]: a.tran[7964]                                                                                                                                                            
    Out[7]: 
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]], dtype=float32)

    In [8]: b.tran[7964]                                                                                                                                                            
    Out[8]: 
    array([[1.38, 0.  , 0.  , 0.  ],
           [0.  , 1.38, 0.  , 0.  ],
           [0.  , 0.  , 1.  , 0.  ],
           [0.  , 0.  , 0.  , 1.  ]], dtype=float32)

    In [9]: a.tran[7965]                                                                                                                                                            
    Out[9]: 
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]], dtype=float32)

    In [10]: b.tran[7965]                                                                                                                                                           
    Out[10]: 
    array([[1.38, 0.  , 0.  , 0.  ],
           [0.  , 1.38, 0.  , 0.  ],
           [0.  , 0.  , 1.  , 0.  ],
           [0.  , 0.  , 0.  , 1.  ]], dtype=float32)

    In [11]: a.tran[7966]                                                                                                                                                           
    Out[11]: 
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]], dtype=float32)

    In [12]: b.tran[7966]                                                                                                                                                           
    Out[12]: 
    array([[1.391, 0.   , 0.   , 0.   ],
           [0.   , 1.391, 0.   , 0.   ],
           [0.   , 0.   , 1.   , 0.   ],
           [0.   , 0.   , 0.   , 1.   ]], dtype=float32)

    In [13]: a.tran[7967]                                                                                                                                                           
    Out[13]: 
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]], dtype=float32)

    In [14]: b.tran[7967]                                                                                                                                                           
    Out[14]: 
    array([[1.391, 0.   , 0.   , 0.   ],
           [0.   , 1.391, 0.   , 0.   ],
           [0.   , 0.   , 1.   , 0.   ],
           [0.   , 0.   , 0.   , 1.   ]], dtype=float32)

    In [15]: a.tran[7986]                                                                                                                                                           
    Out[15]: 
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]], dtype=float32)

    In [16]: b.tran[7986]                                                                                                                                                           
    Out[16]: 
    array([[1.346, 0.   , 0.   , 0.   ],
           [0.   , 1.346, 0.   , 0.   ],
           [0.   , 0.   , 1.   , 0.   ],
           [0.   , 0.   , 0.   , 1.   ]], dtype=float32)

    In [17]:                                                

