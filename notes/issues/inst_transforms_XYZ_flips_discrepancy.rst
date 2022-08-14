inst_transforms_XYZ_flips_discrepancy
=======================================

Previous :doc:`sensor_info_into_new_workflow` showed getting inst identity info to match.
BUT: now getting mismatch between the transforms. 

Generate the geometry and grab using ntds3::

    cd ~/opticks/sysrap/tests
    ./stree_test.sh ana

    010 export STBASE=/tmp/$USER/opticks/ntds3/G4CXOpticks
     14 export FOLD=$STBASE/stree
     15 export CFBASE=$STBASE


    In [1]: cf.inst.view(np.int32)[:,:,3]
    Out[1]: 
    array([[     0,      0,     -1,     -1],
           [     1,      1, 300000,  17612],
           [     2,      1, 300001,  17613],
           [     3,      1, 300002,  17614],
           [     4,      1, 300003,  17615],
           ...,
           [ 48472,      9,     -1,     -1],
           [ 48473,      9,     -1,     -1],
           [ 48474,      9,     -1,     -1],
           [ 48475,      9,     -1,     -1],
           [ 48476,      9,     -1,     -1]], dtype=int32)

    In [2]: f.inst_f4.view(np.int32)[:,:,3]
    Out[2]: 
    array([[     0,      0,     -1,     -1],
           [     1,      1, 300000,  17612],
           [     2,      1, 300001,  17613],
           [     3,      1, 300002,  17614],
           [     4,      1, 300003,  17615],
           ...,
           [ 48472,      9,     -1,     -1],
           [ 48473,      9,     -1,     -1],
           [ 48474,      9,     -1,     -1],
           [ 48475,      9,     -1,     -1],
           [ 48476,      9,     -1,     -1]], dtype=int32)

    In [3]: np.all( cf.inst.view(np.int32)[:,:,3] == f.inst_f4.view(np.int32)[:,:,3] )
    Out[3]: True





Random sampling ~10 transforms, shows they all differ in X,Y or Z flips. 


Transforms not matching::

    In [37]: f.inst_f4[-1]
    Out[37]: 
    array([[     0.  ,      1.  ,      0.  ,      0.  ],
           [    -1.  ,      0.  ,      0.  ,      0.  ],
           [     0.  ,      0.  ,      1.  ,       nan],
           [-22672.5 ,   6711.2 ,  26504.15,       nan]], dtype=float32)

    In [38]: cf.inst[-1]
    Out[38]: 
    array([[    0.  ,     1.  ,     0.  ,     0.  ],
           [    1.  ,     0.  ,     0.  ,     0.  ],
           [    0.  ,     0.  ,     1.  ,      nan],
           [22672.5 ,  6711.2 , 26504.15,      nan]], dtype=float32)


Clear the identity info, and apply the transform. Shows have X or Y sign flip diffs::

    In [52]: a_inst = cf.inst.copy() 
    In [53]: b_inst = f.inst_f4.copy()        

    In [54]: a_inst[:,:,3] = [0,0,0,1]
    In [55]: b_inst[:,:,3] = [0,0,0,1]

    In [56]: np.dot( np.array([0,0,0,1], dtype=np.float32 ), a_inst[10] )
    Out[56]: array([ 2694.681,  2773.886, 18994.307,     1.   ], dtype=float32)

    In [57]: np.dot( np.array([0,0,0,1], dtype=np.float32 ), b_inst[10] )
    Out[57]: array([ 2694.681, -2773.886, 18994.307,     1.   ], dtype=float32)

    In [62]: np.dot( np.array([0,0,0,1], dtype=np.float32 ), b_inst[-1] )
    Out[62]: array([-22672.5 ,   6711.2 ,  26504.15,      1.  ], dtype=float32)

    In [63]: np.dot( np.array([0,0,0,1], dtype=np.float32 ), a_inst[-1] )
    Out[63]: array([22672.5 ,  6711.2 , 26504.15,     1.  ], dtype=float32)


    In [64]: np.dot( np.array([0,0,0,1], dtype=np.float32 ), a_inst[1000] )
    Out[64]: array([ 8272.514, 16920.074,  4584.33 ,     1.   ], dtype=float32)

    In [65]: np.dot( np.array([0,0,0,1], dtype=np.float32 ), b_inst[1000] )
    Out[65]: array([ -8272.514, -16920.074,  -4584.33 ,      1.   ], dtype=float32)


Hmm : to debug this need to see the transform stack being used in both cases.::

    In [70]: np.all( cf.inst.view(np.int32)[:,:,3]  == f.inst_f4.view(np.int32)[:,:,3] )
    Out[70]: True

    In [71]: iid = cf.inst.view(np.int32)[:,:,3]

    In [75]: iid
    Out[75]: 
    array([[     0,      0,     -1,     -1],
           [     1,      1, 300000,  17612],
           [     2,      1, 300001,  17613],
           [     3,      1, 300002,  17614],
           [     4,      1, 300003,  17615],
           ...,
           [ 48472,      9,     -1,     -1],
           [ 48473,      9,     -1,     -1],
           [ 48474,      9,     -1,     -1],
           [ 48475,      9,     -1,     -1],
           [ 48476,      9,     -1,     -1]], dtype=int32)

    In [78]: np.all( iid[:,0] == np.arange(len(iid)) )   ## 1st column is ins_idx
    Out[78]: True

    In [77]: iid[np.where( iid[:,1] == 2 )]
    Out[77]: 
    array([[25601,     2,     2,     2],
           [25602,     2,     4,     4],
           [25603,     2,     6,     6],
           [25604,     2,    21,    21],
           [25605,     2,    22,    22],
           ...,
           [38211,     2, 17586, 17586],
           [38212,     2, 17587, 17587],
           [38213,     2, 17588, 17588],
           [38214,     2, 17589, 17589],
           [38215,     2, 17590, 17590]], dtype=int32)

    In [81]: iid[np.where( iid[:,1] == 3 )]
    Out[81]: 
    array([[38216,     3,     0,     0],
           [38217,     3,     1,     1],
           [38218,     3,     3,     3],
           [38219,     3,     5,     5],
           [38220,     3,     7,     7],
           ...,
           [43208,     3, 17607, 17607],
           [43209,     3, 17608, 17608],
           [43210,     3, 17609, 17609],
           [43211,     3, 17610, 17610],
           [43212,     3, 17611, 17611]], dtype=int32)

    In [82]: a_inst[38216]
    Out[82]: 
    array([[    1.   ,     0.   ,     0.   ,     0.   ],
           [    0.   ,     1.   ,     0.   ,     0.   ],
           [    0.   ,     0.   ,     1.   ,     0.   ],
           [  930.298,   111.872, 19365.   ,     1.   ]], dtype=float32)

    In [83]: b_inst[38216]
    Out[83]: 
    array([[   -1.   ,     0.   ,    -0.   ,     0.   ],
           [    0.   ,     1.   ,     0.   ,     0.   ],
           [    0.   ,     0.   ,    -1.   ,     0.   ],
           [ -930.298,  -111.872, 19365.   ,     1.   ]], dtype=float32)


    In [84]: np.dot( np.array([0,0,0,1], dtype=np.float32 ), a_inst[38216] )
    Out[84]: array([  930.298,   111.872, 19365.   ,     1.   ], dtype=float32)

    In [85]: np.dot( np.array([0,0,0,1], dtype=np.float32 ), b_inst[38216] )
    Out[85]: array([ -930.298,  -111.872, 19365.   ,     1.   ], dtype=float32)


    In [89]: origin = np.array([0,0,0,1], dtype=np.float32 )

    In [92]: ii = 38216
    In [93]: ii, np.dot( origin, a_inst[ii] ), np.dot( origin, b_inst[ii] ) 
    Out[93]: 
    (38216,
     array([  930.298,   111.872, 19365.   ,     1.   ], dtype=float32),
     array([ -930.298,  -111.872, 19365.   ,     1.   ], dtype=float32))

    In [96]: ii, np.dot( origin, a_inst[ii] ), np.dot( origin, b_inst[ii] )
    Out[96]: 
    (48472,
     array([20133.6  ,  9250.101, 26489.85 ,     1.   ], dtype=float32),
     array([-20133.6  ,   9250.101,  26489.85 ,      1.   ], dtype=float32))



    In [97]: a_inst[40000]
    Out[97]: 
    array([[    0.138,     0.254,     0.957,     0.   ],
           [    0.879,     0.477,     0.   ,     0.   ],
           [    0.457,     0.841,     0.29 ,     0.   ],
           [ 8881.754, 16344.179,  5626.955,     1.   ]], dtype=float32)

    In [98]: b_inst[40000]
    Out[98]: 
    array([[   -0.138,    -0.254,     0.957,     0.   ],
           [   -0.879,     0.477,     0.   ,     0.   ],
           [   -0.457,    -0.841,    -0.29 ,     0.   ],
           [ 8881.754, 16344.179,  5626.955,     1.   ]], dtype=float32)

    In [100]: ii=40000 ; ii, np.dot( origin, a_inst[ii] ), np.dot( origin, b_inst[ii] )
    Out[100]: 
    (40000,
     array([ 8881.754, 16344.179,  5626.955,     1.   ], dtype=float32),
     array([ 8881.754, 16344.179,  5626.955,     1.   ], dtype=float32))

    In [101]: plus_z = np.array( [0,0,100,1], dtype=np.float32 )

    In [102]: ii=40000 ; ii, np.dot( plus_z, a_inst[ii] ), np.dot( plus_z, b_inst[ii] )
    Out[102]: 
    (40000,
     array([ 8927.456, 16428.28 ,  5655.909,     1.   ], dtype=float32),
     array([ 8836.052, 16260.078,  5598.001,     1.   ], dtype=float32))



Where the transforms come from
---------------------------------





