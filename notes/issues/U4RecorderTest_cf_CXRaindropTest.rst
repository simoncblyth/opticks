U4RecorderTest_cf_CXRaindropTest : low stat non-aligned comparisons of U4RecorderTest and CXRaindropTest : finding big issues
================================================================================================================================

* from :doc:`U4RecorderTest-shakedown`

Doing simple A-B comparisons with::

    cx
    ./cxs_raindrop.sh       # on workstation 

    u4
    cd tests
    ./U4RecorderTest.sh     # on laptop

    cx 
    ./cxs_raindrop.sh grab  # on laptop
    
    cd ~/opticks/u4/tests 
    ./U4RecorderTest_ab.sh  # on laptop



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






normal incidence b polz unchanging, a does a bit
---------------------------------------------------

::

    In [4]: a.record[0,:4]                                                                                                                                                        
    Out[4]: 
    array([[[  -0.774,   -0.245,    0.583,    0.1  ],
            [  -0.774,   -0.245,    0.583,    1.   ],
            [  -0.602,    0.   ,   -0.799,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[ -38.712,  -12.26 ,   29.173,    0.326],
            [  -0.774,   -0.245,    0.583,    0.   ],
            [  -0.544,    0.009,   -0.839,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[-100.   ,  -31.67 ,   75.357,    0.59 ],
            [  -0.774,   -0.245,    0.583,    0.   ],
            [  -0.544,    0.009,   -0.839,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]]], dtype=float32)

    In [5]: b.record[0,:4]                                                                                                                                                        
    Out[5]: 
    array([[[  -0.774,   -0.245,    0.583,    0.1  ],
            [  -0.774,   -0.245,    0.583,    0.   ],
            [  -0.602,    0.   ,   -0.799,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[ -38.712,  -12.26 ,   29.173,    0.325],
            [  -0.774,   -0.245,    0.583,    0.   ],
            [  -0.602,    0.   ,   -0.799,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[-100.   ,  -31.67 ,   75.357,    0.689],
            [  -0.774,   -0.245,    0.583,    0.   ],
            [  -0.602,    0.   ,   -0.799,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]]], dtype=float32)

    In [6]:                                                            



Pos and mom are close, apart from one BR bouncer
--------------------------------------------------

::

    In [5]: a.photon[:,0]                                                                                                                                                         
    Out[5]: 
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

    In [6]: b.photon[:,0]                                                                                                                                                         
    Out[6]: 
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

    In [7]: a.photon[:,1]                                                                                                                                                         
    Out[7]: 
    array([[-0.774, -0.245,  0.583,  0.   ],
           [-0.217, -0.975,  0.058,  0.   ],
           [-0.791, -0.596,  0.136,  0.   ],
           [-0.504, -0.146,  0.851,  0.   ],
           [-0.456,  0.237, -0.858,  0.   ],
           [ 0.343,  0.448,  0.826,  0.   ],
           [-0.26 ,  0.108, -0.96 ,  0.   ],
           [ 0.581, -0.469,  0.665,  0.   ],
           [ 0.809, -0.188,  0.556,  0.   ],
           [-0.5  ,  0.45 ,  0.74 ,  0.   ]], dtype=float32)

    In [8]: b.photon[:,1]                                                                                                                                                         
    Out[8]: 
    array([[-0.774, -0.245,  0.583,  0.   ],
           [-0.217, -0.975,  0.058,  0.   ],
           [-0.791, -0.596,  0.136,  0.   ],
           [-0.504, -0.146,  0.851,  0.   ],
           [-0.456,  0.237, -0.858,  0.   ],
           [-0.343, -0.448, -0.826,  0.   ],
           [-0.26 ,  0.108, -0.96 ,  0.   ],
           [ 0.581, -0.469,  0.665,  0.   ],
           [ 0.809, -0.188,  0.556,  0.   ],
           [-0.5  ,  0.45 ,  0.74 ,  0.   ]], dtype=float32)


polz very different::

    In [12]: a.photon[:,2]                                                                                                                                                        
    Out[12]: 
    array([[ -0.544,   0.009,  -0.839, 440.   ],
           [ -0.258,   0.   ,  -0.966, 440.   ],
           [  0.179,  -0.457,  -0.871, 440.   ],
           [  0.757,   0.404,   0.513, 440.   ],
           [  0.883,   0.   ,  -0.469, 440.   ],
           [  0.878,  -0.42 ,   0.228, 440.   ],
           [  0.969,  -0.02 ,  -0.245, 440.   ],
           [ -0.753,   0.   ,   0.658, 440.   ],
           [ -0.566,   0.   ,   0.824, 440.   ],
           [ -0.256,  -0.948,   0.19 , 440.   ]], dtype=float32)

    In [13]: b.photon[:,2]                                                                                                                                                        
    Out[13]: 
    array([[ -0.774,  -0.245,   0.583, 440.   ],
           [ -0.217,  -0.975,   0.058, 440.   ],
           [ -0.791,  -0.596,   0.136, 440.   ],
           [ -0.504,  -0.146,   0.851, 440.   ],
           [ -0.456,   0.237,  -0.858, 440.   ],
           [ -0.343,  -0.448,  -0.826, 440.   ],
           [ -0.26 ,   0.108,  -0.96 , 440.   ],
           [  0.581,  -0.469,   0.665, 440.   ],
           [  0.809,  -0.188,   0.556, 440.   ],
           [ -0.5  ,   0.45 ,   0.74 , 440.   ]], dtype=float32)


Huh geant4 giving mom and pol the same, maybe trivial recording bug:: 

    In [17]: a.record[1,:4]                                                                                                                                                       
    Out[17]: 
    array([[[  -0.217,   -0.975,    0.058,    0.2  ],
            [  -0.217,   -0.975,    0.058,    1.   ],
            [  -0.258,    0.   ,   -0.966,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[ -10.831,  -48.727,    2.889,    0.426],
            [  -0.217,   -0.975,    0.058,    0.   ],
            [  -0.258,    0.   ,   -0.966,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[ -22.228, -100.   ,    5.93 ,    0.602],
            [  -0.217,   -0.975,    0.058,    0.   ],
            [  -0.258,    0.   ,   -0.966,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]]], dtype=float32)

    In [18]: b.record[1,:4]                                                                                                                                                       
    Out[18]: 
    array([[[  -0.217,   -0.975,    0.058,    0.2  ],
            [  -0.217,   -0.975,    0.058,    0.   ],
            [  -0.217,   -0.975,    0.058,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[ -10.831,  -48.727,    2.889,    0.425],
            [  -0.217,   -0.975,    0.058,    0.   ],
            [  -0.217,   -0.975,    0.058,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[ -22.228, -100.   ,    5.93 ,    0.667],
            [  -0.217,   -0.975,    0.058,    0.   ],
            [  -0.217,   -0.975,    0.058,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]]], dtype=float32)

    In [19]:                                                                


Does not look like a trivial issue. So perhaps normal incidence handling difference?::

     34 void U4StepPoint::Update(sphoton& photon, const G4StepPoint* point)  // static
     35 {   
     36     const G4ThreeVector& pos = point->GetPosition();
     37     const G4ThreeVector& mom = point->GetMomentumDirection();
     38     const G4ThreeVector& pol = point->GetPolarization();
     39     
     40     G4double time = point->GetGlobalTime();
     41     G4double energy = point->GetKineticEnergy();
     42     G4double wavelength = h_Planck*c_light/energy ;
     43     
     44     photon.pos.x = pos.x();
     45     photon.pos.y = pos.y();
     46     photon.pos.z = pos.z(); 
     47     photon.time  = time/ns ;
     48     
     49     photon.mom.x = mom.x();
     50     photon.mom.y = mom.y();
     51     photon.mom.z = mom.z();
     52     //photon.iindex = 0u ; 
     53     
     54     photon.pol.x = pol.x();
     55     photon.pol.y = pol.y();
     56     photon.pol.z = pol.z(); 
     57     photon.wavelength = wavelength/nm ;
     58 }


FIXED Trivial polz input_photon bug on input, not output recording::

     49 template<typename P>
     50 inline void U4VPrimaryGenerator::GetPhotonParam(
     51      G4ThreeVector& position_mm, G4double& time_ns,
     52      G4ThreeVector& direction,  G4double& wavelength_nm,
     53      G4ThreeVector& polarization, const P& p )
     54 {    
     55      position_mm.set(p.pos.x, p.pos.y, p.pos.z);
     56      time_ns = p.time ;
     57      
     58      direction.set(p.mom.x, p.mom.y, p.mom.z ); 
     59      polarization.set(p.mom.x, p.mom.y, p.mom.z );
       ^^^^^^^^^^^ OOPS ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     60      wavelength_nm = p.wavelength ;
     61 }
     62 



TODO: debug deep dive Geant4 at normal incidence to understand the polz are getting
--------------------------------------------------------------------------------------

::

    cd ~/opticks/u4/tests
    BP=G4OpBoundaryProcess::DielectricDielectric ./U4RecorderTest.sh dbg 


g4-cls G4OpBoundaryProcess


::

    1140               if (sint1 > 0.0) {
    1141                  A_trans = OldMomentum.cross(theFacetNormal);
    1142                  A_trans = A_trans.unit();
    1143                  E1_perp = OldPolarization * A_trans;
    1144                  E1pp    = E1_perp * A_trans;
    1145                  E1pl    = OldPolarization - E1pp;
    1146                  E1_parl = E1pl.mag();
    1147               }
    1148               else {
    1149                  A_trans  = OldPolarization;
    1150                  // Here we Follow Jackson's conventions and we set the
    1151                  // parallel component = 1 in case of a ray perpendicular
    1152                  // to the surface
    1153                  E1_perp  = 0.0;
    1154                  E1_parl  = 1.0;
    1155               }
    1156 
    1157               s1 = Rindex1*cost1;
    1158               E2_perp = 2.*s1*E1_perp/(Rindex1*cost1+Rindex2*cost2);
    1159               E2_parl = 2.*s1*E1_parl/(Rindex2*cost1+Rindex1*cost2);
    1160               E2_total = E2_perp*E2_perp + E2_parl*E2_parl;
    1161               s2 = Rindex2*cost2*E2_total;
    1162 




FIXED : cx 2/10 with nan polz
--------------------------------
::

     670 inline QSIM_METHOD int qsim::propagate_at_boundary(unsigned& flag, sphoton& p, const quad2* prd, const qstate& s, curandStateXORWOW& rng, unsigned idx)
     671 {
     672     const float& n1 = s.material1.x ;
     673     const float& n2 = s.material2.x ;
     674     const float eta = n1/n2 ;
     675 
     676     const float3* normal = (float3*)&prd->q0.f.x ;
     677 
     678     const float _c1 = -dot(p.mom, *normal );
     679     const float3 oriented_normal = _c1 < 0.f ? -(*normal) : (*normal) ;
     680     const float3 trans = cross(p.mom, oriented_normal) ;
     681     const float trans_length = length(trans) ;
     682     const float c1 = fabs(_c1) ;
     683     const bool normal_incidence = trans_length == 0.f  ;
     684 
     685     /**
     686     **Normal Incidence**
     687  
     688     Judging normal_incidence based on absolete dot product being exactly unity "c1 == 1.f" is problematic 
     689     as when very near to normal incidence there are vectors for which the absolute dot product 
     690     is not quite 1.f but the cross product does give an exactly zero vector which gives 
     691     A_trans (nan, nan, nan) from the normalize doing : (zero,zero,zero)/zero.   
     692 
     693     Solution is to judge normal incidence based on trans_length as that is what the 
     694     calulation actually needs to be non-zero in order to be able to calculate A_trans.
     695     Hence should be able to guarantee that A_trans will be well defined. 
     696     **/
     697 




After fix::

    N[blyth@localhost CSGOptiX]$ PIDX=1 ./cxs_raindrop.sh 
    ..

    //qsim.propagate idx 1 bnc 0 cosTheta     1.0000 dir (   -0.2166    -0.9745     0.0578) nrm (   -0.2166    -0.9745     0.0578) 
    //qsim.propagate idx 1 bounce 0 command 3 flag 0 s.optical.x 0 
    //qsim.propagate_at_boundary idx 1 nrm   (    0.2166     0.9745    -0.0578) 
    //qsim.propagate_at_boundary idx 1 mom_0 (   -0.2166    -0.9745     0.0578) 
    //qsim.propagate_at_boundary idx 1 pol_0 (   -0.2578     0.0000    -0.9662) 
    //qsim.propagate_at_boundary idx 1 c1     1.0000 normal_incidence 1 
    //qsim.propagate_at_boundary idx 1 normal_incidence 1 p.pol (   -0.2578,    0.0000,   -0.9662) p.mom (   -0.2166,   -0.9745,    0.0578) o_normal (    0.2166,    0.9745,   -0.0578)
    //qsim.propagate_at_boundary idx 1 TransCoeff     0.9775 n1c1     1.3530 n2c2     1.0003 E2_t (    0.0000,    1.1499) A_trans (   -0.2578,    0.0000,   -0.9662) 
    //qsim.propagate_at_boundary idx 1 reflect 0 tir 0 TransCoeff     0.9775 u_reflect     0.3725 
    //qsim.propagate_at_boundary idx 1 mom_1 (   -0.2166    -0.9745     0.0578) 
    //qsim.propagate_at_boundary idx 1 pol_1 (   -0.2578     0.0000    -0.9662) 
    //qsim.propagate idx 1 bnc 1 cosTheta     0.9745 dir (   -0.2166    -0.9745     0.0578) nrm (    0.0000    -1.0000     0.0000) 
    //qsim.propagate idx 1 bounce 1 command 3 flag 0 s.optical.x 99 
    2022-06-15 03:19:39.793 INFO  [432148] [SEvt::save@944] DefaultDir /tmp/blyth/opticks/GeoChain/BoxedSphere/CXRaindropTest
    2022-06-15 03:19:39.793 INFO  [432148] [SEvt::save@970]  dir /tmp/blyth/opticks/GeoChain/BoxedSphere/CXRaindropTest
    2022-06-15 03:19:39.793 INFO  [432148] [QEvent::getPhoton@345] [ evt.num_photon 10 p.sstr (10, 4, 4, ) evt.photon 0x7f88d8000000


PIDX dumping::

    N[blyth@localhost CSGOptiX]$ PIDX=1 ./cxs_raindrop.sh 

    //qsim.propagate idx 1 bnc 0 cosTheta     1.0000 dir (   -0.2166    -0.9745     0.0578) nrm (   -0.2166    -0.9745     0.0578) 
    //qsim.propagate idx 1 bounce 0 command 3 flag 0 s.optical.x 0 
    //qsim.propagate_at_boundary idx 1 nrm   (    0.2166     0.9745    -0.0578) 
    //qsim.propagate_at_boundary idx 1 mom_0 (   -0.2166    -0.9745     0.0578) 
    //qsim.propagate_at_boundary idx 1 pol_0 (   -0.2578     0.0000    -0.9662) 
    //qsim.propagate_at_boundary idx 1 c1     1.0000 normal_incidence 0 
    //qsim.propagate_at_boundary idx 1 reflect 0 tir 0 TransCoeff        nan u_reflect     0.3725 
    //qsim.propagate_at_boundary idx 1 mom_1 (   -0.2166    -0.9745     0.0578) 
    //qsim.propagate_at_boundary idx 1 pol_1 (       nan        nan        nan) 
    //qsim.propagate idx 1 bnc 1 cosTheta     0.9745 dir (   -0.2166    -0.9745     0.0578) nrm (    0.0000    -1.0000     0.0000) 
    //qsim.propagate idx 1 bounce 1 command 3 flag 0 s.optical.x 99 
    2022-06-15 02:08:59.420 INFO  [426728] [SEvt::save@944] DefaultDir /tmp/blyth/opticks/GeoChain/BoxedSphere/CXRaindropTest
    2022-06-15 02:08:59.420 INFO  [426728] [SEvt::save@970]  dir /tmp/blyth/opticks/GeoChain/BoxedSphere/CXRaindropTest
    2022-06-15 02:08:59.420 INFO  [426728] [QEvent::getPhoton@345] [ evt.num_photon 10 p.sstr (10, 4, 4, ) evt.photon 0x7f8ef8000000


Issue is from cross product with very close to normal incidence but not quite::

    //qsim.propagate_at_boundary idx 1 pol_0 (   -0.2578     0.0000    -0.9662) 
    //qsim.propagate_at_boundary idx 1 c1     1.0000 normal_incidence 0 
    //qsim.propagate_at_boundary idx 1 normal_incidence 0 p.pol (   -0.2578,    0.0000,   -0.9662) p.mom (   -0.2166,   -0.9745,    0.0578) o_normal (    0.2166,    0.9745,   -0.0578)
    //qsim.propagate_at_boundary idx 1 TransCoeff        nan n1c1     1.3530 n2c2     1.0003 E2_t (       nan,       nan) A_trans (       nan,       nan,       nan) 
    //qsim.propagate_at_boundary idx 1 reflect 0 tir 0 TransCoeff        nan u_reflect     0.3725 


::

    539 /** cross product */
     540 SUTIL_INLINE SUTIL_HOSTDEVICE float3 cross(const float3& a, const float3& b)
     541 {
     542   return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
     543 }

     552 SUTIL_INLINE SUTIL_HOSTDEVICE float3 normalize(const float3& v)
     553 {
     554   float invLen = 1.0f / sqrtf(dot(v, v));
     555   return v * invLen;
     556 }






ana/input_photons.py

    214     @classmethod
    215     def GenerateRandomSpherical(cls, n):
    216         """
    217         spherical distribs not carefully checked  
    218 
    219         The start position is offset by the direction vector for easy identification purposes
    220         so that means the rays will start on a virtual unit sphere and travel radially 
    221         outwards from there.
    222 
    223         """

Dumping normals, looks as expected. cosTheta 1 means the rays all exit the sphere in radial direction.::

    //qsim.propagate idx 0 bnc 0 cosTheta     1.0000 dir (   -0.7742    -0.2452     0.5835) nrm (   -0.7742    -0.2452     0.5835) 
    //qsim.propagate idx 1 bnc 0 cosTheta     1.0000 dir (   -0.2166    -0.9745     0.0578) nrm (   -0.2166    -0.9745     0.0578) 
    //qsim.propagate idx 2 bnc 0 cosTheta     1.0000 dir (   -0.7913    -0.5961     0.1361) nrm (   -0.7913    -0.5961     0.1361) 
    //qsim.propagate idx 3 bnc 0 cosTheta     1.0000 dir (   -0.5041    -0.1461     0.8512) nrm (   -0.5041    -0.1461     0.8512) 
    //qsim.propagate idx 4 bnc 0 cosTheta     1.0000 dir (   -0.4558     0.2371    -0.8579) nrm (   -0.4558     0.2371    -0.8579) 
    //qsim.propagate idx 5 bnc 0 cosTheta     1.0000 dir (   -0.3432    -0.4476    -0.8257) nrm (   -0.3432    -0.4476    -0.8257) 
    //qsim.propagate idx 6 bnc 0 cosTheta     1.0000 dir (   -0.2601     0.1076    -0.9596) nrm (   -0.2601     0.1076    -0.9596) 
    //qsim.propagate idx 7 bnc 0 cosTheta     1.0000 dir (    0.5806    -0.4695     0.6652) nrm (    0.5806    -0.4695     0.6652) 
    //qsim.propagate idx 8 bnc 0 cosTheta     1.0000 dir (    0.8094    -0.1881     0.5563) nrm (    0.8094    -0.1881     0.5563) 
    //qsim.propagate idx 9 bnc 0 cosTheta     1.0000 dir (   -0.5001     0.4497     0.7400) nrm (   -0.5001     0.4497     0.7400) 
    //qsim.propagate idx 0 bnc 1 cosTheta     0.7742 dir (   -0.7742    -0.2452     0.5835) nrm (   -1.0000     0.0000     0.0000) 
    //qsim.propagate idx 1 bnc 1 cosTheta     0.9745 dir (   -0.2166    -0.9745     0.0578) nrm (    0.0000    -1.0000     0.0000) 
    //qsim.propagate idx 2 bnc 1 cosTheta     0.7913 dir (   -0.7913    -0.5961     0.1361) nrm (   -1.0000     0.0000     0.0000) 
    //qsim.propagate idx 3 bnc 1 cosTheta     0.8512 dir (   -0.5041    -0.1461     0.8512) nrm (    0.0000     0.0000     1.0000) 
    //qsim.propagate idx 4 bnc 1 cosTheta     0.8579 dir (   -0.4558     0.2371    -0.8579) nrm (    0.0000     0.0000    -1.0000) 

    //qsim.propagate idx 5 bnc 1 cosTheta     1.0000 dir (    0.3432     0.4476     0.8257) nrm (    0.3432     0.4476     0.8257) 
    HMM:  TO BR BT SA

    //qsim.propagate idx 6 bnc 1 cosTheta     0.9596 dir (   -0.2601     0.1076    -0.9596) nrm (    0.0000     0.0000    -1.0000) 
    //qsim.propagate idx 7 bnc 1 cosTheta     0.6652 dir (    0.5806    -0.4695     0.6652) nrm (    0.0000     0.0000     1.0000) 
    //qsim.propagate idx 8 bnc 1 cosTheta     0.8094 dir (    0.8094    -0.1881     0.5563) nrm (    1.0000     0.0000     0.0000) 
    //qsim.propagate idx 9 bnc 1 cosTheta     0.7400 dir (   -0.5001     0.4497     0.7400) nrm (    0.0000     0.0000     1.0000) 
    //qsim.propagate idx 5 bnc 2 cosTheta     0.8257 dir (    0.3432     0.4476     0.8257) nrm (    0.0000     0.0000     1.0000) 




::

    In [59]: a.photon[:,2]                                                                                                                                                      
    Out[59]: 
    array([[ -0.544,   0.009,  -0.839, 440.   ],
           [    nan,     nan,     nan, 440.   ],
           [  0.179,  -0.457,  -0.871, 440.   ],
           [  0.757,   0.404,   0.513, 440.   ],
           [    nan,     nan,     nan, 440.   ],
           [  0.923,  -0.337,   0.183, 440.   ],
           [  0.965,   0.   ,  -0.262, 440.   ],
           [ -0.753,   0.   ,   0.658, 440.   ],
           [ -0.566,   0.   ,   0.824, 440.   ],
           [ -0.256,  -0.948,   0.19 , 440.   ]], dtype=float32)




    In [43]: a.record[1,:4]                                                                                                                                                     
    Out[43]: 
    array([[[  -0.217,   -0.975,    0.058,    0.2  ],
            [  -0.217,   -0.975,    0.058,    1.   ],
            [  -0.258,    0.   ,   -0.966,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[ -10.831,  -48.727,    2.889,    0.426],
            [  -0.217,   -0.975,    0.058,    0.   ],
            [     nan,      nan,      nan,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[ -22.228, -100.   ,    5.93 ,    0.602],
            [  -0.217,   -0.975,    0.058,    0.   ],
            [     nan,      nan,      nan,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]]], dtype=float32)

    In [58]: a.record[4,:4]                                                                                                                                                     
    Out[58]: 
    array([[[  -0.456,    0.237,   -0.858,    0.5  ],
            [  -0.456,    0.237,   -0.858,    1.   ],
            [   0.883,    0.   ,   -0.469,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[ -22.789,   11.855,  -42.896,    0.726],
            [  -0.456,    0.237,   -0.858,    0.   ],
            [     nan,      nan,      nan,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[ -53.126,   27.637, -100.   ,    0.948],
            [  -0.456,    0.237,   -0.858,    0.   ],
            [     nan,      nan,      nan,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]]], dtype=float32)





FIXED : cx genflag zeros : in qsim.h::generate_photon
-----------------------------------------------------------

* input photons need to get givenTORCH genflag 
* correct place to do in qsim::generate_photon

::

    192 static __forceinline__ __device__ void simulate( const uint3& launch_idx, const uint3& dim, quad2* prd )
    193 {
    194     sevent* evt      = params.evt ;
    195     if (launch_idx.x >= evt->num_photon) return;
    196 
    197     unsigned idx = launch_idx.x ;  // aka photon_id
    198     unsigned genstep_id = evt->seed[idx] ;
    199     const quad6& gs     = evt->genstep[genstep_id] ;
    200 
    201     qsim* sim = params.sim ;
    202     curandState rng = sim->rngstate[idx] ;    // TODO: skipahead using an event_id 
    203 
    204     sphoton p = {} ;
    205 
    206     sim->generate_photon(p, rng, gs, idx, genstep_id );
    207 


::

    In [1]: seqhis_(a.seq[:,0])                                                                                                                                                 
    Out[1]: 
    ['TO BT SA',
     'TO BT SA',
     'TO BT SA',
     'TO BT SA',
     'TO BT SA',
     'TO BR BT SA',
     'TO BT SA',
     'TO BT SA',
     'TO BT SA',
     'TO BT SA']




::

    In [10]: seqhis_(a.seq[:,0])                                                                                                                                                
    Out[10]: 
    ['?0? BT SA',
     '?0? BT SA',
     '?0? BT SA',
     '?0? BT SA',
     '?0? BT SA',
     '?0? BR BT SA',
     '?0? BT SA',
     '?0? BT SA',
     '?0? BT SA',
     '?0? BT SA']

    In [11]: seqhis_(b.seq[:,0])                                                                                                                                                
    Out[11]: 
    ['TO BT SA',
     'TO BT SA',
     'TO BT SA',
     'TO BT SA',
     'TO BT SA',
     'TO BT SA',
     'TO BT SA',
     'TO BT SA',
     'TO BT SA',
     'TO BT SA']





FIXED : cx missing seq : by using SEventConfig::SetStandardFullDebug
------------------------------------------------------------------------

::

    35 const char* SEventConfig::_CompMaskDefault = SComp::ALL_ ;

    038 struct SYSRAP_API SComp
     39 {
     40     static constexpr const char* ALL_ = "genstep,photon,record,rec,seq,seed,hit,simtrace,domain,inphoton" ;
     41     static constexpr const char* UNDEFINED_ = "undefined" ;
     42     static constexpr const char* GENSTEP_   = "genstep" ;


::

    2022-06-14 22:18:07.758 INFO  [386951] [SEvt::save@944] DefaultDir /tmp/blyth/opticks/GeoChain/BoxedSphere/CXRaindropTest
    2022-06-14 22:18:07.758 INFO  [386951] [SEvt::save@970]  dir /tmp/blyth/opticks/GeoChain/BoxedSphere/CXRaindropTest
    2022-06-14 22:18:07.758 INFO  [386951] [QEvent::getPhoton@345] [ evt.num_photon 10 p.sstr (10, 4, 4, ) evt.photon 0x7f75ec000000
    2022-06-14 22:18:07.758 INFO  [386951] [QEvent::getPhoton@348] ] evt.num_photon 10
    2022-06-14 22:18:07.758 INFO  [386951] [QEvent::getRecord@404]  evt.num_record 100
    2022-06-14 22:18:07.758 INFO  [386951] [QEvent::getRec@411]  getRec called when there is no such array, use SEventConfig::SetCompMask to avoid 
    2022-06-14 22:18:07.758 INFO  [386951] [QEvent::getSeq@388]  getSeq called when there is no such array, use SEventConfig::SetCompMask to avoid 
    2022-06-14 22:18:07.761 INFO  [386951] [QEvent::getHit@479]  evt.photon 0x7f75ec000000 evt.num_photon 10 evt.num_hit 0 selector.hitmask 64 SEventConfig::HitMask 64 SEventConfig::HitMaskLabel SD
    2022-06-14 22:18:07.761 INFO  [386951] [QEvent::getSimtrace@370]  getSimtrace called when there is no such array, use SEventConfig::SetCompMask to avoid 
    2022-06-14 22:18:07.761 INFO  [386951] [SEvt::save@974] SEvt::descComponent
     SEventConfig::CompMaskLabel genstep,photon,record,rec,seq,seed,hit,simtrace,domain,inphoton
                     hit                    - 
                    seed               (10, ) 
                 genstep          (1, 6, 4, )       SEventConfig::MaxGenstep             1000000
                  photon         (10, 4, 4, )        SEventConfig::MaxPhoton             3000000
                  record     (10, 10, 4, 4, )        SEventConfig::MaxRecord                  10
                     rec                    -           SEventConfig::MaxRec                   0
                     seq                    -           SEventConfig::MaxSeq                   0
                  domain          (2, 4, 4, ) 
                simtrace                    - 

    2022-06-14 22:18:07.761 INFO  [386951] [SEvt::save@975] NPFold::desc
                                 genstep.npy : (1, 6, 4, )
                                  photon.npy : (10, 4, 4, )
                                  record.npy : (10, 10, 4, 4, )
                                    seed.npy : (10, )
                                  domain.npy : (2, 4, 4, )
                                inphoton.npy : (10, 4, 4, )


::

    249 bool QEvent::hasSeq() const    { return evt->seq != nullptr ; }

    377 void QEvent::getSeq(NP* seq) const
    378 {
    379     if(!hasSeq()) return ;
    380     LOG(LEVEL) << "[ evt.num_seq " << evt->num_seq << " seq.sstr " << seq->sstr() << " evt.seq " << evt->seq ;
    381     assert( seq->has_shape(evt->num_seq, 2) );
    382     QU::copy_device_to_host<sseq>( (sseq*)seq->bytes(), evt->seq, evt->num_seq );
    383     LOG(LEVEL) << "] evt.num_seq " << evt->num_seq  ;
    384 }



The defaults are all zero for debug records::

     17 int SEventConfig::_MaxRecordDefault = 0 ;
     18 int SEventConfig::_MaxRecDefault = 0 ;
     19 int SEventConfig::_MaxSeqDefault = 0 ;

And cxs_raindrop.sh only upped that for RECORD, now added REC and SEQ::

     91 unset GEOM                     # MUST unset GEOM for CSGFoundry::Load_ to load OPTICKS_KEY basis geometry 
     92 export OPTICKS_MAX_RECORD=10   # change from default of 0, see sysrap/SEventConfig.cc
     93 export OPTICKS_MAX_SEQ=10
     94 export OPTICKS_MAX_REC=10
     95 

From U4RecorderTest::

    164     unsigned max_bounce = 9 ;
    165     SEventConfig::SetMaxBounce(max_bounce);
    166     SEventConfig::SetMaxRecord(max_bounce+1);
    167     SEventConfig::SetMaxRec(max_bounce+1);
    168     SEventConfig::SetMaxSeq(max_bounce+1);


Consolidate to make it easier for debug executables to use same config settings::

    void SEventConfig::SetStandardFullDebug() // static
    {
        unsigned max_bounce = 9 ; 
        SEventConfig::SetMaxBounce(max_bounce); 
        SEventConfig::SetMaxRecord(max_bounce+1); 
        SEventConfig::SetMaxRec(max_bounce+1); 
        SEventConfig::SetMaxSeq(max_bounce+1); 
    }





::

    a.base:/tmp/blyth/opticks/GeoChain/BoxedSphere/CXRaindropTest

      : a.genstep                                          :            (1, 6, 4) : 0:27:47.278953 
      : a.seed                                             :                (10,) : 0:27:47.276945 
      : a.record_meta                                      :                    1 : 0:27:47.277345 
      : a.NPFold_meta                                      :                    2 : 0:27:47.280458 
      : a.record                                           :       (10, 10, 4, 4) : 0:27:47.277733 
      : a.domain                                           :            (2, 4, 4) : 0:27:47.279858 
      : a.inphoton                                         :           (10, 4, 4) : 0:27:47.278531 
      : a.NPFold_index                                     :                    6 : 0:27:47.281013 
      : a.photon                                           :           (10, 4, 4) : 0:27:47.278158 
      : a.domain_meta                                      :                    2 : 0:27:47.279315 

     min_stamp : 2022-06-14 15:47:50.299234 
     max_stamp : 2022-06-14 15:47:50.303302 
     dif_stamp : 0:00:00.004068 
     age_stamp : 0:27:47.276945 

    In [37]: b                                                                                                                                                                  
    Out[37]: 
    b

    CMDLINE:/Users/blyth/opticks/u4/tests/U4RecorderTest_ab.py
    b.base:/tmp/blyth/opticks/U4RecorderTest

      : b.genstep                                          :            (1, 6, 4) : 0:21:56.990119 
      : b.seq                                              :              (10, 2) : 0:21:56.988098 
      : b.record_meta                                      :                    1 : 0:21:56.989270 
      : b.pho0                                             :              (10, 4) : 0:21:56.985779 
      : b.rec_meta                                         :                    1 : 0:21:56.988635 
      : b.rec                                              :       (10, 10, 2, 4) : 0:21:56.988532 
      : b.record                                           :       (10, 10, 4, 4) : 0:21:56.989174 
      : b.domain                                           :            (2, 4, 4) : 0:21:56.986951 
      : b.inphoton                                         :           (10, 4, 4) : 0:21:56.986110 
      : b.pho                                              :              (10, 4) : 0:21:56.985578 
      : b.NPFold_index                                     :                    7 : 0:21:56.990755 
      : b.photon                                           :           (10, 4, 4) : 0:21:56.989561 
      : b.gs                                               :               (1, 4) : 0:21:56.985400 
      : b.domain_meta                                      :                    2 : 0:21:56.987080 

     min_stamp : 2022-06-14 15:53:42.157865 
     max_stamp : 2022-06-14 15:53:42.163220 




