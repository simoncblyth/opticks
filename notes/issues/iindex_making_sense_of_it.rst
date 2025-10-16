iindex_making_sense_of_it
============================



iindex review, prior to sphoton.h change packing orient_iindex together
------------------------------------------------------------------------

TEST=ref1 cxs_min.sh
~~~~~~~~~~~~~~~~~~~~~~~

::

    In [2]: ii = f.photon[:,1,3].view(np.uint32)

    In [3]: ii.min(),ii.max()
    Out[3]: (np.uint32(0), np.uint32(48593))

    In [4]: np.c_[np.unique(ii, return_counts=True)]
    Out[4]: 
    array([[     0, 327770],
           [     2,      2],
           [     4,      3],
           [     5,      1],
           [    10,      2],
           ...,
           [ 48236,     11],
           [ 48237,     13],
           [ 48238,     16],
           [ 48239,     11],
           [ 48593,  66842]], shape=(36477, 2))

    In [5]: f.photon.shape
    Out[5]: (1000000, 4, 4)

    In [6]: f.hit.shape
    Out[6]: (200397, 4, 4)

    In [7]: ii.shape
    Out[7]: (1000000,)

    In [8]: u_ii, n_ii = np.unique(ii, return_counts=True)

    In [9]: u_ii
    Out[9]: array([    0,     2,     4,     5,    10, ..., 48236, 48237, 48238, 48239, 48593], shape=(36477,), dtype=uint32)

    In [10]: n_ii
    Out[10]: array([327770,      2,      3,      1,      2, ...,     11,     13,     16,     11,  66842], shape=(36477,))

    In [11]: np.where(n_ii > 1000)
    Out[11]: (array([    0, 36476]),)




iindex is the index of the OptixInstance in the IAS
-------------------------------------------------------

::

    693 extern "C" __global__ void __closesthit__ch()
    694 {
    695     unsigned iindex = optixGetInstanceIndex() ;
    696     unsigned identity = optixGetInstanceId() ;


So the values of iindex and what the geometry they correspond to depends on the order of the inst qat4 vector::

    0425 void SBT::createIAS(unsigned ias_idx)
     426 {
     427     unsigned num_inst = foundry->getNumInst();
     428     unsigned num_ias_inst = foundry->getNumInstancesIAS(ias_idx, emm);
     429     LOG(LEVEL)
     430         << " ias_idx " << ias_idx
     431         << " num_inst " << num_inst
     432         << " num_ias_inst(getNumInstancesIAS) " << num_ias_inst
     433         ;
     434 
     435     std::vector<qat4> inst ;
     436     foundry->getInstanceTransformsIAS(inst, ias_idx, emm );
     437     assert( num_ias_inst == inst.size() );
     438 
     439 
     440     collectInstances(inst);
     441 


That inst qat4 vector hails from the stree.h inst_f4 vector of transforms that is persisted with the tree::

    In [1]: f.inst_f4.shape
    Out[1]: (48594, 4, 4)


iindex:"-1" must be the triangulated global solid with identity transform::

    In [4]: f.inst_f4[48593]
    Out[4]: 
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  1., nan],
           [ 0.,  0.,  0., nan]], dtype=float32)

    In [5]: f.inst_f4[48593].view(np.uint32)
    Out[5]: 
    array([[1065353216,          0,          0,      48593],
           [         0, 1065353216,          0,         11],
           [         0,          0, 1065353216, 4294967295],
           [         0,          0,          0, 4294967295]], dtype=uint32)


iindex:0 is the analytic global solid::

    In [6]: 0xffffffff
    Out[6]: 4294967295

    In [7]: f.inst_f4[0].view(np.uint32)
    Out[7]: 
    array([[1065353216,          0,          0,          0],
           [         0, 1065353216,          0,          0],
           [         0,          0, 1065353216, 4294967295],
           [         0,          0,          0, 4294967295]], dtype=uint32)


::

    5365 inline void stree::add_inst(
    5366     glm::tmat4x4<double>& tr_m2w,
    5367     glm::tmat4x4<double>& tr_w2m,
    5368     int gas_idx,
    5369     int nidx )
    5370 {
    5371     assert( nidx > -1 && nidx < int(nds.size()) );
    5372     const snode& nd = nds[nidx];    // structural volume node
    5373 
    5374     int ins_idx = int(inst.size()); // 0-based index follow sqat4.h::setIdentity 
    5375 
    5376     glm::tvec4<int64_t> col3 ;   // formerly uint64_t
    5377 
    5378     col3.x = ins_idx ;            // formerly  +1
    5379     col3.y = gas_idx ;            // formerly  +1
    5380     col3.z = nd.sensor_id ;       // formerly ias_idx + 1 (which was always 1)
    5381     col3.w = nd.sensor_index ;
    5382 
    5383     strid::Encode(tr_m2w, col3 );
    5384     strid::Encode(tr_w2m, col3 );
    5385 
    5386     inst.push_back(tr_m2w);
    5387     iinst.push_back(tr_w2m);
    5388 
    5389     inst_nidx.push_back(nidx);
    5390 }



That 11 is the gas_idx for the tri global compound solid::

    (ok) A[blyth@localhost CSGFoundry]$ cat.py mmlabel.txt
    0    2554:sWorld
    1    5:PMT_3inch_pmt_solid
    2    9:NNVTMCPPMTsMask_virtual
    3    12:HamamatsuR12860sMask_virtual
    4    4:mask_PMT_20inch_vetosMask_virtual
    5    1:sStrutBallhead
    6    1:base_steel
    7    3:uni_acrylic1
    8    130:sPanel
    9    1:sStrut_0
    10   6:PMT_20inch_pmt_solid_head
    11   338:ConnectingCutTube_0
    (ok) A[blyth@localhost CSGFoundry]$ 




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



