Geant4_using_GROUPVEL_from_wrong_initial_material_after_refraction
=====================================================================

* From :doc:`U4RecorderTest_cf_CXRaindropTest`


Geant4 Time Is Wrong Until *UseGivenVelocity*
------------------------------------------------ 

After adding UseGivenVelocity::

    void U4Recorder::PreUserTrackingAction_Optical(const G4Track* track)
    {
    +    const_cast<G4Track*>(track)->UseGivenVelocity(true);  



SMOKING GUN : AB check of dist/time and velocity between points 0,1 and 2 reveals that G4 propagating like Water in what should be Air
----------------------------------------------------------------------------------------------------------------------------------------

final photon (pos,t)::

    In [1]: a.photon[:,0]                                                                                                                                                         
    Out[1]: 
    array([[-100.   ,  -31.67 ,   75.357,    0.59 ],
           [ -22.228, -100.   ,    5.93 ,    0.602],
           [-100.   ,  -75.341,   17.199,    0.781],
           [ -59.225,  -17.159,  100.   ,    0.851],
           [ -53.126,   27.637, -100.   ,    0.948],
           [  41.563,   54.208,  100.   ,    1.525],
           [ -27.109,   11.211, -100.   ,    1.107],
           [  87.27 ,  -70.573,  100.   ,    1.361],
           [ 100.   ,  -23.237,   68.731,    1.372],
           [ -67.583,   60.769,  100.   ,    1.51 ]], dtype=float32)

    In [2]: b.photon[:,0]                                                                                                                                                         
    Out[2]: 
    array([[-100.   ,  -31.67 ,   75.357,    0.689],
           [ -22.228, -100.   ,    5.93 ,    0.667],
           [-100.   ,  -75.341,   17.199,    0.876],
           [ -59.225,  -17.159,  100.   ,    0.935],
           [ -53.126,   27.637, -100.   ,    1.031],
           [ -41.563,  -54.208, -100.   ,    1.152],
           [ -27.109,   11.211, -100.   ,    1.174],
           [  87.27 ,  -70.573,  100.   ,    1.486],
           [ 100.   ,  -23.237,   68.731,    1.463],
           [ -67.583,   60.769,  100.   ,    1.616]], dtype=float32)


After adding UseGivenVelocity::

    void U4Recorder::PreUserTrackingAction_Optical(const G4Track* track)
    {
    +    const_cast<G4Track*>(track)->UseGivenVelocity(true);  

    In [4]: a.photon[:,0]
    Out[4]: 
    array([[-100.   ,  -31.67 ,   75.357,    0.59 ],
           [ -22.228, -100.   ,    5.93 ,    0.602],
           [-100.   ,  -75.341,   17.199,    0.781],
           [ -59.225,  -17.159,  100.   ,    0.851],
           [ -53.126,   27.637, -100.   ,    0.948],
           [  41.563,   54.208,  100.   ,    1.525],
           [ -27.109,   11.211, -100.   ,    1.107],
           [  87.27 ,  -70.573,  100.   ,    1.361],
           [ 100.   ,  -23.237,   68.731,    1.372],
           [ -67.583,   60.769,  100.   ,    1.51 ]], dtype=float32)


    In [3]: b.photon[:,0]
    Out[3]: 
    array([[-100.   ,  -31.67 ,   75.357,    0.589],
           [ -22.228, -100.   ,    5.93 ,    0.601],
           [-100.   ,  -75.341,   17.199,    0.78 ],
           [ -59.225,  -17.159,  100.   ,    0.85 ],
           [ -53.126,   27.637, -100.   ,    0.947],
           [ -41.563,  -54.208, -100.   ,    1.062],
           [ -27.109,   11.211, -100.   ,    1.106],
           [  87.27 ,  -70.573,  100.   ,    1.36 ],
           [ 100.   ,  -23.237,   68.731,    1.371],
           [ -67.583,   60.769,  100.   ,    1.509]], dtype=float32)





First boundary (pos,t) is close::

    In [11]: a.record[:,1,0]
    Out[11]: 
    array([[-38.712, -12.26 ,  29.173,   0.326],
           [-10.831, -48.727,   2.889,   0.426],
           [-39.563, -29.807,   6.804,   0.526],
           [-25.206,  -7.303,  42.56 ,   0.626],
           [-22.789,  11.855, -42.896,   0.726],
           [-17.16 , -22.381, -41.287,   0.826],
           [-13.006,   5.379, -47.978,   0.926],
           [ 29.028, -23.474,  33.262,   1.026],
           [ 40.47 ,  -9.404,  27.816,   1.126],
           [-25.007,  22.485,  37.001,   1.226]], dtype=float32)

    In [12]: b.record[:,1,0]
    Out[12]: 
    array([[-38.712, -12.26 ,  29.173,   0.325],
           [-10.831, -48.727,   2.889,   0.425],
           [-39.563, -29.807,   6.804,   0.525],
           [-25.206,  -7.303,  42.56 ,   0.625],
           [-22.789,  11.855, -42.896,   0.725],
           [-17.16 , -22.381, -41.287,   0.825],
           [-13.006,   5.379, -47.978,   0.925],
           [ 29.028, -23.474,  33.262,   1.025],
           [ 40.47 ,  -9.404,  27.816,   1.125],
           [-25.007,  22.485,  37.001,   1.225]], dtype=float32)


Second boundary pos (all TO BT SA other than idx 5 : TO BR BT SA) on cube faces except 5::

    In [35]: a.record[:,2,0,:3]
    Out[35]: 
    array([[-100.   ,  -31.67 ,   75.357],
           [ -22.228, -100.   ,    5.93 ],
           [-100.   ,  -75.341,   17.199],
           [ -59.225,  -17.159,  100.   ],
           [ -53.126,   27.637, -100.   ],
              [  17.16 ,   22.381,   41.287],    ## 5
           [ -27.109,   11.211, -100.   ],
           [  87.27 ,  -70.573,  100.   ],
           [ 100.   ,  -23.237,   68.731],
           [ -67.583,   60.769,  100.   ]], dtype=float32)

    In [36]: b.record[:,2,0,:3]
    Out[36]: 
    array([[-100.   ,  -31.67 ,   75.357],
           [ -22.228, -100.   ,    5.93 ],
           [-100.   ,  -75.341,   17.199],
           [ -59.225,  -17.159,  100.   ],
           [ -53.126,   27.637, -100.   ],
           [ -41.563,  -54.208, -100.   ],
           [ -27.109,   11.211, -100.   ],
           [  87.27 ,  -70.573,  100.   ],
           [ 100.   ,  -23.237,   68.731],
           [ -67.583,   60.769,  100.   ]], dtype=float32)


Time difference at 2nd bounce suggests using different material props::

    In [37]: a.record[:,2,0,3]
    Out[37]: array([0.59 , 0.602, 0.781, 0.851, 0.948, 1.288, 1.107, 1.361, 1.372, 1.51 ], dtype=float32)

    In [38]: b.record[:,2,0,3]
    Out[38]: array([0.689, 0.667, 0.876, 0.935, 1.031, 1.152, 1.174, 1.486, 1.463, 1.616], dtype=float32)

    In [39]: b.record[:,2,0,3] - a.record[:,2,0,3]
    Out[39]: array([ 0.098,  0.065,  0.095,  0.084,  0.083, -0.136,  0.067,  0.125,  0.091,  0.106], dtype=float32)


Expected 1mm radius at starting position (RandomSpherical10_f8.npy input photons)::

    In [33]: np.sqrt(np.sum( b.record[:,0,0,:3]*b.record[:,0,0,:3], axis=1 ))
    Out[33]: array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=float32)

    In [34]: np.sqrt(np.sum( a.record[:,0,0,:3]*a.record[:,0,0,:3], axis=1 ))
    Out[34]: array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=float32)

Expected 50mm radius at first boundary::

    In [31]: np.sqrt(np.sum( a.record[:,1,0,:3]*a.record[:,1,0,:3], axis=1 ))
    Out[31]: array([50., 50., 50., 50., 50., 50., 50., 50., 50., 50.], dtype=float32)

    In [32]: np.sqrt(np.sum( b.record[:,1,0,:3]*b.record[:,1,0,:3], axis=1 ))
    Out[32]: array([50., 50., 50., 50., 50., 50., 50., 50., 50., 50.], dtype=float32)

Expected point 0->1 distance of 49mm::

    In [47]: a_d01 = a.record[:,1,0,:3] - a.record[:,0,0,:3]
    In [49]: np.sqrt(np.sum( a_d01*a_d01, axis=1 ))
    Out[49]: array([49., 49., 49., 49., 49., 49., 49., 49., 49., 49.], dtype=float32)

    In [50]: b_d01 = b.record[:,1,0,:3] - b.record[:,0,0,:3]
    In [51]: np.sqrt(np.sum(b_d01*b_d01, axis=1 )) 
    Out[51]: array([49., 49., 49., 49., 49., 49., 49., 49., 49., 49.], dtype=float32)

    In [52]: a_dist_ = lambda i:np.sqrt(np.sum( (a.record[:,i+1,0,:3]-a.record[:,i,0,:3])*(a.record[:,i+1,0,:3]-a.record[:,i,0,:3]) , axis=1 ))
    In [53]: a_dist_(0)
    Out[53]: array([49., 49., 49., 49., 49., 49., 49., 49., 49., 49.], dtype=float32)

    In [54]: b_dist_ = lambda i:np.sqrt(np.sum( (b.record[:,i+1,0,:3]-b.record[:,i,0,:3])*(b.record[:,i+1,0,:3]-b.record[:,i,0,:3]) , axis=1 ))
    In [55]: b_dist_(0)
    Out[55]: array([49., 49., 49., 49., 49., 49., 49., 49., 49., 49.], dtype=float32)


Point 1->2 distance::

    In [56]: a_dist_(1)   ## the 100. is diameter across the sphere of the TO BR->BT SA
    Out[56]: array([ 79.157,  52.612,  76.381,  67.482,  66.56 , 100.   ,  54.214, 100.322,  73.547,  85.131], dtype=float32)

    In [57]: b_dist_(1)
    Out[57]: array([ 79.157,  52.612,  76.381,  67.482,  66.56 ,  71.103,  54.214, 100.322,  73.547,  85.131], dtype=float32)

Time between consecutive record points::

    In [58]: a_time_ = lambda i:a.record[:,i+1,0,3] - a.record[:,i,0,3]
    In [59]: b_time_ = lambda i:b.record[:,i+1,0,3] - b.record[:,i,0,3]


Time between 0 and 1::

    In [60]: a_time_(0)
    Out[60]: array([0.226, 0.226, 0.226, 0.226, 0.226, 0.226, 0.226, 0.226, 0.226, 0.226], dtype=float32)

    In [61]: b_time_(0)
    Out[61]: array([0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225], dtype=float32)

Speed mm/ns between 0 and 1::

    In [66]: a_dist_(0)/a_time_(0)
    Out[66]: array([216.601, 216.601, 216.601, 216.601, 216.601, 216.601, 216.601, 216.601, 216.601, 216.601], dtype=float32)

    In [67]: b_dist_(0)/b_time_(0)
    Out[67]: array([217.658, 217.658, 217.658, 217.658, 217.658, 217.658, 217.658, 217.658, 217.658, 217.658], dtype=float32)

Time between 1 and 2::

    In [62]: a_time_(1)
    Out[62]: array([0.264, 0.176, 0.255, 0.225, 0.222, 0.462, 0.181, 0.335, 0.245, 0.284], dtype=float32)

    In [63]: b_time_(1)
    Out[63]: array([0.364, 0.242, 0.351, 0.31 , 0.306, 0.327, 0.249, 0.461, 0.338, 0.391], dtype=float32)

Speed mm/ns between 1 and 2::

    In [68]: a_dist_(1)/a_time_(1)                              [..Water.]
    Out[68]: array([299.712, 299.712, 299.711, 299.712, 299.712, 216.601, 299.711, 299.712, 299.712, 299.712], dtype=float32)

    In [69]: b_dist_(1)/b_time_(1)    ## HUH: getting same speed as with water in whay should be air ?
    Out[69]: array([217.658, 217.658, 217.658, 217.658, 217.658, 217.658, 217.658, 217.658, 217.658, 217.658], dtype=float32)

First boundary time : G4 0.001 ns earlier::

    In [21]: a.record[:,1,0,3]
    Out[21]: array([0.326, 0.426, 0.526, 0.626, 0.726, 0.826, 0.926, 1.026, 1.126, 1.226], dtype=float32)

    In [22]: b.record[:,1,0,3]
    Out[22]: array([0.325, 0.425, 0.525, 0.625, 0.725, 0.825, 0.925, 1.025, 1.125, 1.225], dtype=float32)

    In [23]: a.record[:,1,0,3]/b.record[:,1,0,3]
    Out[23]: array([1.003, 1.003, 1.002, 1.002, 1.002, 1.001, 1.001, 1.001, 1.001, 1.001], dtype=float32)

    In [24]: a.record[:,1,0,3]-b.record[:,1,0,3]
    Out[24]: array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], dtype=float32)


Have Vague Recollection of a Similar Time Problem Before : related to the GROUPVEL ?
----------------------------------------------------------------------------------------

* something about calculate the velocity ?

::

    414 CCtx::setTrackOptical
    415 --------------------------
    416 
    417 Invoked by CCtx::setTrack
    418 
    419 
    420 UseGivenVelocity(true)
    421 ~~~~~~~~~~~~~~~~~~~~~~~~
    422 
    423 NB without this BoundaryProcess proposed velocity to get correct GROUPVEL for material after refraction 
    424 are trumpled by G4Track::CalculateVelocity 
    425 
    426 **/
    427 
    428 void CCtx::setTrackOptical(G4Track* mtrack)
    429 {
    430     mtrack->UseGivenVelocity(true);
    431 
    432     _pho = CPhotonInfo::Get(mtrack);


::

    epsilon:issues blyth$ opticks-f UseGivenVelocity
    ./cfg4/CCtx.cc:UseGivenVelocity(true)
    ./cfg4/CCtx.cc:    mtrack->UseGivenVelocity(true);
    ./cfg4/DsG4OpBoundaryProcess.cc:    //     G4Track::UseGivenVelocity is in force, that is done in CTrackingAction
    ./examples/Geant4/CerenkovMinimal/src/Ctx.cc:    const_cast<G4Track*>(track)->UseGivenVelocity(true);
    epsilon:opticks blyth$ 

    epsilon:issues blyth$ find . -name '*.rst' -exec grep -H UseGivenVelocity {} \;
    ./strategy_for_Cerenkov_Scintillation_alignment.rst:     69     const_cast<G4Track*>(track)->UseGivenVelocity(true);
    ./strategy_for_Cerenkov_Scintillation_alignment.rst:    241     LOG(debug) << "CTrackingAction::setTrack setting UseGivenVelocity for optical " ;
    ./strategy_for_Cerenkov_Scintillation_alignment.rst:    243     _track->UseGivenVelocity(true);
    ./geant4_opticks_integration/GROUPVEL.rst:After G4Track::UseGivenVelocity requiring a const_cast in CTrackingAction get the correct velocities and times::
    ./CRecorder_record_id_ni_assert_CAUSED_BY_DsG4Scintillation_INSTRUMENTATION_REMOVED.rst:    397     mtrack->UseGivenVelocity(true);
    ./U4RecorderTest_cf_CXRaindropTest.rst:    420 UseGivenVelocity(true)
    ./U4RecorderTest_cf_CXRaindropTest.rst:    430     mtrack->UseGivenVelocity(true);
    ./U4RecorderTest_cf_CXRaindropTest.rst:    epsilon:issues blyth$ opticks-f UseGivenVelocity
    ./U4RecorderTest_cf_CXRaindropTest.rst:    ./cfg4/CCtx.cc:UseGivenVelocity(true)
    ./U4RecorderTest_cf_CXRaindropTest.rst:    ./cfg4/CCtx.cc:    mtrack->UseGivenVelocity(true);
    ./U4RecorderTest_cf_CXRaindropTest.rst:    ./cfg4/DsG4OpBoundaryProcess.cc:    //     G4Track::UseGivenVelocity is in force, that is done in CTrackingAction
    ./U4RecorderTest_cf_CXRaindropTest.rst:    ./examples/Geant4/CerenkovMinimal/src/Ctx.cc:    const_cast<G4Track*>(track)->UseGivenVelocity(true);
    ./reemission_review.rst:    405     LOG(debug) << "CTrackingAction::setTrack setting UseGivenVelocity for optical " ;
    ./reemission_review.rst:    407     _track->UseGivenVelocity(true);
    ./reemission_review.rst:    392     LOG(debug) << "CTrackingAction::setTrack setting UseGivenVelocity for optical " ;
    ./reemission_review.rst:    394     _track->UseGivenVelocity(true);
    epsilon:issues blyth$ 
    epsilon:issues blyth$ 


Quoting from notes/issues/geant4_opticks_integration/GROUPVEL.rst::

    Dumping from DebugG4Navigation::

        2016-11-21 22:31:05.318 INFO  [1546020] [CMaterialLib::dumpGroupvelMaterial@38]   5     trans.ASDIP.beg nm   430 nm/ns    194.519 ns    15.3969 lkp GdDopedLS qwn 
        2016-11-21 22:31:05.318 INFO  [1546020] [CMaterialLib::dumpGroupvelMaterial@38]   0     trans.ASDIP.beg nm   430 nm/ns    194.519 ns  0.0514088 lkp GdDopedLS qwn 
        2016-11-21 22:31:05.318 INFO  [1546020] [CMaterialLib::dumpGroupvelMaterial@38]   1     trans.ASDIP.beg nm   430 nm/ns     192.78 ns     5.1354 lkp Acrylic qwn 
        2016-11-21 22:31:05.319 INFO  [1546020] [CMaterialLib::dumpGroupvelMaterial@38]   2     trans.ASDIP.beg nm   430 nm/ns    194.519 ns  0.0514088 lkp GdDopedLS qwn 
        2016-11-21 22:31:05.319 INFO  [1546020] [CMaterialLib::dumpGroupvelMaterial@38]   3     trans.ASDIP.beg nm   430 nm/ns     192.78 ns     5.1354 lkp Acrylic qwn 

    After G4Track::UseGivenVelocity requiring a const_cast in CTrackingAction get the correct velocities and times::

        2016-11-21 22:46:59.837 INFO  [1549372] [CMaterialLib::dumpGroupvelMaterial@38]   5     trans.ASDIP.beg nm   430 nm/ns    194.519 ns    15.3969 lkp GdDopedLS qwn 
        2016-11-21 22:46:59.837 INFO  [1549372] [CMaterialLib::dumpGroupvelMaterial@38]   0     trans.ASDIP.beg nm   430 nm/ns     192.78 ns  0.0518727 lkp Acrylic qwn 
        2016-11-21 22:46:59.837 INFO  [1549372] [CMaterialLib::dumpGroupvelMaterial@38]   1     trans.ASDIP.beg nm   430 nm/ns    194.519 ns    5.08947 lkp GdDopedLS qwn 
        2016-11-21 22:46:59.838 INFO  [1549372] [CMaterialLib::dumpGroupvelMaterial@38]   2     trans.ASDIP.beg nm   430 nm/ns     192.78 ns  0.0518727 lkp Acrylic qwn 
        2016-11-21 22:46:59.838 INFO  [1549372] [CMaterialLib::dumpGroupvelMaterial@38]   3     trans.ASDIP.beg nm   430 nm/ns    197.134 ns    5.02196 lkp MineralOil qwn 




