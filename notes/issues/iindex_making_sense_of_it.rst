iindex_making_sense_of_it
============================

HMM : how to make sense of the iindex values ?
-------------------------------------------------


A temporary fix::

    a.f.record[:,:,1,3][np.where( a.f.record[:,:,1,3] == 1. )] = 0. 


::

    In [10]: ii = a.f.record[:,:,1,3].view(np.int32)

    In [11]: ii.min()
    Out[11]: 0

    In [12]: ii.max()
    Out[12]: 47966

::

    In [14]: cf.inst.shape
    Out[14]: (48477, 4, 4)


::

    uii = np.c_[np.unique(ii, return_counts=True )] 


    In [23]: uii[uii[:,1]>1000]
    Out[23]: 
    array([[      0, 2587492],
           [  17337,    3240],
           [  17820,    2936],
           [  28212,    2085],
           [  39124,    1889],
           [  39216,  572529]])

    In [28]: sii = uii[uii[:,1]>1000][1:,0] ; sii
    Out[28]: array([17337, 17820, 28212, 39124, 39216])




    In [29]: cf.inst[sii]
    Out[29]: 
    array([[[     0.49 ,     -0.386,      0.782,      0.   ],
            [    -0.619,     -0.785,      0.   ,      0.   ],
            [     0.614,     -0.484,     -0.624,      0.   ],
            [-11893.05 ,   9384.445,  12092.436,      0.   ]],

           [[     0.469,     -0.37 ,      0.802,      0.   ],
            [    -0.619,     -0.785,      0.   ,      0.   ],
            [     0.629,     -0.497,     -0.598,      0.   ],
            [-12200.587,   9627.113,  11584.637,      0.   ]],

           [[     0.45 ,     -0.355,      0.819,      0.   ],
            [    -0.619,     -0.785,      0.   ,      0.   ],
            [     0.643,     -0.507,     -0.574,      0.   ],
            [-12496.27 ,   9860.413,  11148.806,      0.   ]],

           [[     0.509,     -0.401,      0.762,      0.   ],
            [    -0.619,     -0.785,      0.   ,      0.   ],
            [     0.598,     -0.472,     -0.648,      0.   ],
            [-11623.11 ,   9171.431,  12588.428,      0.   ]],

           [[     0.48 ,     -0.379,      0.792,      0.   ],
            [    -0.619,     -0.785,      0.   ,      0.   ],
            [     0.621,     -0.49 ,     -0.611,      0.   ],
            [-12075.873,   9528.691,  11876.771,      0.   ]]], dtype=float32)

    In [31]: cf.inst[sii][:,:,3].view(np.int32)
    Out[31]: 
    array([[ 17337,      1, 317337,  34948],
           [ 17820,      1, 317820,  35431],
           [ 28212,      2,   3703,   3702],
           [ 39124,      3,   3007,   3006],
           [ 39216,      3,   3355,   3354]], dtype=int32)

::

    375     /**
    376     sqat4::setIdentity
    377     -------------------
    378 
    379     Canonical usage from CSGFoundry::addInstance  where sensor_identifier gets +1 
    380     with 0 meaning not a sensor. 
    381     **/
    382 
    383     QAT4_METHOD void setIdentity(int ins_idx, int gas_idx, int sensor_identifier_1, int sensor_index )
    384     {
    385         assert( sensor_identifier_1 >= 0 );
    386 
    387         q0.i.w = ins_idx ;             // formerly unsigned and "+ 1"
    388         q1.i.w = gas_idx ;
    389         q2.i.w = sensor_identifier_1 ;   // now +1 with 0 meaning not-a-sensor 
    390         q3.i.w = sensor_index ;
    391     }


::

    In [33]: np.c_[cf.mmlabel]
    Out[33]: 
    array([['2977:sWorld'],                              0
           ['5:PMT_3inch_pmt_solid'],                    1
           ['9:NNVTMCPPMTsMask_virtual'],                2
           ['12:HamamatsuR12860sMask_virtual'],          3
           ['6:mask_PMT_20inch_vetosMask_virtual'],      4
           ['1:sStrutBallhead'],
           ['1:uni1'],
           ['1:base_steel'],
           ['1:uni_acrylic1'],
           ['130:sPanel']], dtype=object)



How to pick indices that have no 3inch in their histories ?
-------------------------------------------------------------

::

    In [58]: ii[:,:10]
    Out[58]: 
    array([[    0, 39216, 39216, 39216, 39216, 39216, 39216, 39216,     0, 28212],
           [    0, 39216, 39216, 39216, 17820, 17820,     0,     0,     0,     0],
           [    0, 39216, 39216, 39216, 39216, 39216, 17820, 17820,     0,     0],
           [    0, 39216, 39216, 39216, 39216, 39216, 39216, 39216,     0,     0],
           [    0, 39216, 39216, 39216, 39216, 39216,     0,     0,     0,     0],
           [    0, 39216, 39216, 39216, 39216, 39216, 17820, 17820,     0,     0],
           [    0, 39216, 39216, 39216, 39216, 39216,     0,     0,     0,     0],
           [    0, 39216, 39216, 39216, 39216, 39216, 39216, 39216,     0,     0],
           ...,
           [    0, 39216, 39216, 39216, 39216, 39216,     0,     0,     0,     0],
           [    0, 39216, 39216, 39216, 39216, 39216,     0,     0,     0,     0],
           [    0, 39216, 39216, 39216, 39216, 39216,     0,     0,     0,     0],
           [    0, 39216, 39216, 39216, 39216, 39216, 39216, 39216, 39216, 39216],
           [    0, 39216, 39216, 39216, 39216, 39216,     0,     0,     0,     0],
           [    0, 39216, 39216, 39216, 17337, 17337, 17337,     0,     0,     0],
           [    0, 39216, 39216, 39216, 39216, 39216,     0,     0,     0,     0],
           [    0, 39216, 39216, 39216, 39216, 39216,     0,     0,     0,     0]], dtype=int32)

    In [59]: ii.shape
    Out[59]: (100000, 32)


::

    In [71]: np.unique(ii[1])
    Out[71]: array([    0, 17820, 39216], dtype=int32)

    In [72]: np.unique(ii[2])
    Out[72]: array([    0, 17820, 39216], dtype=int32)

    In [73]: np.unique(ii[1000])
    Out[73]: array([    0, 39216], dtype=int32)

    In [74]: np.unique(ii[1001])
    Out[74]: array([    0, 39216], dtype=int32)



    ii = a.f.record[:,:,1,3].view(np.int32)

    In [94]: w3 = np.unique( np.where( np.logical_or( ii == 17337, ii == 17820)  )[0] )  ; w3
    Out[94]: array([    1,     2,     5,    10,    22,    28,    34,    36, ..., 99965, 99966, 99976, 99978, 99981, 99989, 99991, 99997])


    In [97]: np.c_[np.unique(a.q[w3], return_counts=True)]   ## histories with 3inch involved
    Out[97]: 
    array([[b'TO BT BR BT BT AB                                                                               ', b'10'],
           [b'TO BT BR BT BT BR BR BT BT BT BT BT SR BT BT BT BT DR BT DR AB                                  ', b'1'],
           [b'TO BT BR BT BT BR BT BT BR BT BT BT BT BT SA                                                    ', b'1'],
           [b'TO BT BR BT BT BR BT BT BT BT BT SA                                                             ', b'1'],
           [b'TO BT BR BT BT BR BT BT BT BT BT SR BT BT BT BT DR BT DR AB                                     ', b'2'],
           [b'TO BT BR BT BT BR BT BT BT BT BT SR BT BT BT BT DR BT SA                                        ', b'1'],
           [b'TO BT BR BT BT BR BT BT BT BT SA                                                                ', b'1'],
           [b'TO BT BR BT BT BR BT BT BT SC SC BT BT SA                                                       ', b'1'],
           [b'TO BT BR BT BT BT AB                                                                            ', b'23'],
           [b'TO BT BR BT BT BT BT AB                                                                         ', b'5'],
           [b'TO BT BR BT BT BT BT BR BR BT DR BT DR AB                                                       ', b'1'],
           [b'TO BT BR BT BT BT BT BR BT BT AB                                                                ', b'1'],
           [b'TO BT BR BT BT BT BT BT AB                                                                      ', b'7'],
           [b'TO BT BR BT BT BT BT BT SA                                                                      ', b'12'],
           [b'TO BT BR BT BT BT BT SA                                                                         ', b'2'],
           [b'TO BT BR BT BT BT SA                                                                            ', b'66'],
           [b'TO BT BR BT BT BT SD                                                                            ', b'103'],
           [b'TO BT BR BT BT SA                                                                               ', b'15'],
           [b'TO BT BT BR BR BR BT BT BT AB                                                                   ', b'2'],
           [b'TO BT BT BR BT BT AB                                                                            ', b'1'],
           [b'TO BT BT BR BT BT BT AB                                                                         ', b'179'],
           [b'TO BT BT BR BT BT BT BR BR BT BT BT BT BR DR AB                                                 ', b'1'],
           [b'TO BT BT BR BT BT BT BR BT BT BT AB                                                             ', b'1'],

           ...




::

    In [13]: for v in range(10): print(v, repr(np.where(cfid[:,1] == v )[0]), np.where(cfid[:,1] == v )[0].shape )                                                               
    0 array([0]) (1,)
    1 array([    1,     2,     3,     4,     5, ..., 25596, 25597, 25598, 25599, 25600]) (25600,)
    2 array([25601, 25602, 25603, 25604, 25605, ..., 38211, 38212, 38213, 38214, 38215]) (12615,)
    3 array([38216, 38217, 38218, 38219, 38220, ..., 43208, 43209, 43210, 43211, 43212]) (4997,)
    4 array([43213, 43214, 43215, 43216, 43217, ..., 45608, 45609, 45610, 45611, 45612]) (2400,)
    5 array([45613, 45614, 45615, 45616, 45617, 45618, 45619, 45620, 45621, 45622, 45623, 45624, 45625, 45626, 45627, 45628, 45629, 45630, 45631, 45632, 45633, 45634, 45635, 45636, 45637, 45638, 45639,
           45640, 45641, 45642, 45643, 45644, 45645, 45646, 45647, 45648, 45649, 4



