random_aligned_cerenkov_generation_test
==========================================


qudarap/tests/QCtxTest  QCtxTest::rng_sequence
-------------------------------------------------

qudarap/QCtx.cu::

     15 __global__ void _QCtx_rng_sequence(qctx* ctx, float* rs, unsigned num_items )
     16 {
     17     unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
     18     if (id >= num_items) return;
     19     curandState rng = *(ctx->r + id) ; 
     20     float u = curand_uniform(&rng) ;
     21     if(id % 100000 == 0) printf("//_QCtx_rng_sequence id %d u %10.4f    \n", id, u  );
     22     rs[id] = u ; 
     23 }

* currently just collects the first random float from each photon slot  


thrustrap/tests/TRngBufTest
-----------------------------

* collects 16*16 double randoms for each photon slot

::

    In [1]: a = np.load("/tmp/blyth/opticks/TRngBufTest_0.npy")                                                                                                                                    

    In [2]: a                                                                                                                                                                                      
    Out[2]: 
    array([[[0.74 , 0.438, 0.517, ..., 0.547, 0.653, 0.23 ],
            [0.339, 0.761, 0.546, ..., 0.855, 0.489, 0.189],
            [0.507, 0.021, 0.958, ..., 0.748, 0.488, 0.318],
            ...,
            [0.153, 0.327, 0.894, ..., 0.94 , 0.946, 0.197],
            [0.856, 0.657, 0.063, ..., 0.624, 0.968, 0.532],
            [0.902, 0.429, 0.674, ..., 0.598, 0.82 , 0.145]],

           [[0.921, 0.46 , 0.333, ..., 0.825, 0.527, 0.93 ],
            [0.163, 0.785, 0.942, ..., 0.492, 0.543, 0.934],
            [0.479, 0.449, 0.126, ..., 0.042, 0.379, 0.715],

    In [5]: a.shape                                                                                                                                                                                
    Out[5]: (10000, 16, 16)




compare those
------------------

::

    In [21]: a[:100,0,0]                                                                                                                                                                           
    Out[21]: 
    array([0.74 , 0.921, 0.039, 0.969, 0.925, 0.446, 0.667, 0.11 , 0.47 , 0.513, 0.776, 0.295, 0.714, 0.359, 0.681, 0.292, 0.319, 0.811, 0.154, 0.445, 0.208, 0.611, 0.307, 0.416, 0.234, 0.879, 0.646,
           0.926, 0.579, 0.554, 0.356, 0.723, 0.278, 0.619, 0.588, 0.375, 0.24 , 0.415, 0.094, 0.633, 0.285, 0.779, 0.213, 0.413, 0.033, 0.536, 0.721, 0.355, 0.253, 0.985, 0.92 , 0.187, 0.182, 0.598,
           0.708, 0.042, 0.731, 0.94 , 0.843, 0.612, 0.267, 0.021, 0.833, 0.722, 0.609, 0.63 , 0.53 , 0.813, 0.059, 0.48 , 0.991, 0.879, 1.   , 0.207, 0.437, 0.373, 0.447, 0.238, 0.034, 0.731, 0.494,
           0.303, 0.809, 0.129, 0.783, 0.073, 0.124, 0.223, 0.742, 0.627, 0.153, 0.012, 0.173, 0.478, 0.805, 0.687, 0.302, 0.808, 0.407, 0.751])

    In [22]: r[:100]                                                                                                                                                                               
    Out[22]: 
    array([0.74 , 0.921, 0.039, 0.969, 0.925, 0.446, 0.667, 0.11 , 0.47 , 0.513, 0.776, 0.295, 0.714, 0.359, 0.681, 0.292, 0.319, 0.811, 0.154, 0.445, 0.208, 0.611, 0.307, 0.416, 0.234, 0.879, 0.646,
           0.926, 0.579, 0.554, 0.356, 0.723, 0.278, 0.619, 0.588, 0.375, 0.24 , 0.415, 0.094, 0.633, 0.285, 0.779, 0.213, 0.413, 0.033, 0.536, 0.721, 0.355, 0.253, 0.985, 0.92 , 0.187, 0.182, 0.598,
           0.708, 0.042, 0.731, 0.94 , 0.843, 0.612, 0.267, 0.021, 0.833, 0.722, 0.609, 0.63 , 0.53 , 0.813, 0.059, 0.48 , 0.991, 0.879, 1.   , 0.207, 0.437, 0.373, 0.447, 0.238, 0.034, 0.731, 0.494,
           0.303, 0.809, 0.129, 0.783, 0.073, 0.124, 0.223, 0.742, 0.627, 0.153, 0.012, 0.173, 0.478, 0.805, 0.687, 0.302, 0.808, 0.407, 0.751], dtype=float32)

    In [23]:              



cerenkov generation check using random alignment
---------------------------------------------------

* getting geant4 to use the same randoms in cks opticks/examples/Geant4/CerenkovStandalone/G4Cerenkov_modifiedTest.cc
  would be real helpful for debugging why the cerenkov wavelength histogram sample matching is poor

* potential cause : float vs double, if so need to drill down as to exactly where
 


Getting G4Cerenkov_modified to use precooked randoms using OpticksRandom
-----------------------------------------------------------------------------

::

    G4Cerenkov_modifiedTest::PSDI [BetaInverse_1.500_step_length_100.000_SKIP_CONTINUE]
     i 0 rand0    0.74022 Pmin/eV    1.55000 Pmax/eV   15.50000 dp    0.00001 sampledEnergy/eV   11.87606 sampledRI    1.45360 cosTheta    1.03192 sin2Theta   -0.06486 rand1    0.43845
     i 0 rand0    0.51701 Pmin/eV    1.55000 Pmax/eV   15.50000 dp    0.00001 sampledEnergy/eV    8.76233 sampledRI    1.68320 cosTheta    0.89116 sin2Theta    0.20583 rand1    0.15699



Use same precooked randoms from python
----------------------------------------

::

    epsilon:CerenkovStandalone blyth$ ipython -i cks.py 
    idx     0 u0    0.74022 sampledEnergy   11.87606 sampledRI    1.45360 cosTheta    1.03192 sin2Theta   -0.06486 u1    0.43845
    idx     0 u0    0.51701 sampledEnergy    8.76233 sampledRI    1.68320 cosTheta    0.89116 sin2Theta    0.20583 u1    0.15699

    idx     1 u0    0.92099 sampledEnergy   14.39786 sampledRI    1.45360 cosTheta    1.03192 sin2Theta   -0.06486 u1    0.46036
    idx     1 u0    0.33346 sampledEnergy    6.20182 sampledRI    1.61849 cosTheta    0.92679 sin2Theta    0.14107 u1    0.37252

    idx     2 u0    0.03902 sampledEnergy    2.09434 sampledRI    1.48406 cosTheta    1.01074 sin2Theta   -0.02160 u1    0.25021
    idx     2 u0    0.18448 sampledEnergy    4.12356 sampledRI    1.52616 cosTheta    0.98286 sin2Theta    0.03399 u1    0.96242
    idx     2 u0    0.52055 sampledEnergy    8.81174 sampledRI    1.67328 cosTheta    0.89644 sin2Theta    0.19639 u1    0.93996
    idx     2 u0    0.83058 sampledEnergy   13.13657 sampledRI    1.45360 cosTheta    1.03192 sin2Theta   -0.06486 u1    0.40973
    idx     2 u0    0.08162 sampledEnergy    2.68863 sampledRI    1.49337 cosTheta    1.00444 sin2Theta   -0.00890 u1    0.80677
    idx     2 u0    0.69529 sampledEnergy   11.24924 sampledRI    1.45360 cosTheta    1.03192 sin2Theta   -0.06486 u1    0.61771
    idx     2 u0    0.25633 sampledEnergy    5.12587 sampledRI    1.57064 cosTheta    0.95502 sin2Theta    0.08793 u1    0.21368

    idx     3 u0    0.96896 sampledEnergy   15.06703 sampledRI    1.45360 cosTheta    1.03192 sin2Theta   -0.06486 u1    0.49474
    idx     3 u0    0.67338 sampledEnergy   10.94366 sampledRI    1.45360 cosTheta    1.03192 sin2Theta   -0.06486 u1    0.56277
    idx     3 u0    0.12019 sampledEnergy    3.22671 sampledRI    1.50301 cosTheta    0.99800 sin2Theta    0.00400 u1    0.97649
    idx     3 u0    0.13583 sampledEnergy    3.44485 sampledRI    1.50864 cosTheta    0.99427 sin2Theta    0.01142 u1    0.58897
    idx     3 u0    0.49062 sampledEnergy    8.39412 sampledRI    1.75709 cosTheta    0.85368 sin2Theta    0.27122 u1    0.32844

    idx     4 u0    0.92514 sampledEnergy   14.45571 sampledRI    1.45360 cosTheta    1.03192 sin2Theta   -0.06486 u1    0.05301
    idx     4 u0    0.16310 sampledEnergy    3.82528 sampledRI    1.51846 cosTheta    0.98784 sin2Theta    0.02417 u1    0.88969



    2021-07-15 20:09:06.370 INFO  [795925] [QCtx::generate_cerenkov_photon@277] [ num_photon 100
    //QCtx_generate_cerenkov_photon num_photon 100 
    //qctx::cerenkov_photon id 0 u0     0.7402 sampledRI     1.4536 cosTheta     1.0319 sin2Theta    -0.0649 u1     0.4385 
    //qctx::cerenkov_photon id 0 u0     0.5170 sampledRI     1.6834 cosTheta     0.8910 sin2Theta     0.2060 u1     0.1570 
    //_QCtx_generate_cerenkov_photon id 0 





    In [26]: np.set_printoptions(precision=6)

    In [27]: a[0,0,:4]
    Out[27]: array([0.740219, 0.438451, 0.517013, 0.156989])


    In [15]: Pmin = 1.55                                                                                                                                                                           

    In [16]: Pmax = 15.5                                                                                                                                                                           

    In [17]: e = Pmin + u0*(Pmax - Pmin)                                                                                                                                                           

    In [18]: e                                                                                                                                                                                     
    Out[18]: 11.876059997081757

    rindex = np.load("/tmp/G4Cerenkov_modifiedTest/BetaInverse_1.500_step_length_100.000_SKIP_CONTINUE/RINDEX.npy") 

    rindex_ = lambda ev:np.interp( ev, rindex[:,0], rindex[:,1] )  


    In [28]: rindex_(11.87606)                                                                                                                                                                     
    Out[28]: 1.4536

    In [29]: 1.5/rindex_(11.87606)                                                                                                                                                                 
    Out[29]: 1.0319207484865163


    In [30]: cosTheta_ = lambda ev:1.5/rindex_(ev)                                                                                                                                                 

    In [31]: cosTheta_(e)                                                                                                                                                                          
    Out[31]: 1.0319207484865163

    In [32]: e                                                                                                                                                                                     
    Out[32]: 11.876059997081757

    In [33]: rindex_(e)                                                                                                                                                                            
    Out[33]: 1.4536

    In [34]: r = rindex_(e) ; r                                                                                                                                                                    
    Out[34]: 1.4536

    In [35]: sin2Theta_ = lambda e:(1.0 - cosTheta_(e))*(1.0 + cosTheta_(e))                                                                                                                       

    In [36]: sin2Theta_(e)                                                                                                                                                                         
    Out[36]: -0.06486043115697197





::

    0258   G4double Pmin = Rindex->GetMinLowEdgeEnergy();
     259   G4double Pmax = Rindex->GetMaxLowEdgeEnergy();


     g4-cls G4MaterialPropertyVector
     g4-cls G4PhysicsOrderedFreeVector


    124 inline
    125 G4double G4PhysicsOrderedFreeVector::GetMaxLowEdgeEnergy()
    126 {
    127   return binVector.back();
    128 }
    129 
    130 inline
    131 G4double G4PhysicsOrderedFreeVector::GetMinLowEdgeEnergy()
    132 {
    133   return binVector.front();
    134 }



    079 void G4PhysicsOrderedFreeVector::InsertValues(G4double energy, G4double value)
     80 {
     81         std::vector<G4double>::iterator binLoc =
     82                  std::lower_bound(binVector.begin(), binVector.end(), energy);
     83 
     84         size_t binIdx = binLoc - binVector.begin(); // Iterator difference!
     85 
     86         std::vector<G4double>::iterator dataLoc = dataVector.begin() + binIdx;
     87 
     88         binVector.insert(binLoc, energy);
     89         dataVector.insert(dataLoc, value);
     90 
     91         ++numberOfNodes;
     92         edgeMin = binVector.front();
     93         edgeMax = binVector.back();
     94 }

     96 G4double G4PhysicsOrderedFreeVector::GetEnergy(G4double aValue)
     97 {
     98         G4double e;
     99         if (aValue <= GetMinValue()) {
    100           e = edgeMin;
    101         } else if (aValue >= GetMaxValue()) {
    102           e = edgeMax;
    103         } else {
    104           size_t closestBin = FindValueBinLocation(aValue);
    105           e = LinearInterpolationOfEnergy(aValue, closestBin);
    106     }
    107         return e;
    108 }


    231 inline
    232  G4double G4PhysicsVector::Value(G4double theEnergy) const
    233 {
    234   size_t idx=0;
    235   return Value(theEnergy, idx);
    236 }
    237 

    498 G4double G4PhysicsVector::Value(G4double theEnergy, size_t& lastIdx) const
    499 {
    500   G4double y;
    501   if(theEnergy <= edgeMin) {
    502     lastIdx = 0;
    503     y = dataVector[0];
    504   } else if(theEnergy >= edgeMax) {
    505     lastIdx = numberOfNodes-1;
    506     y = dataVector[lastIdx];
    507   } else {
    508     lastIdx = FindBin(theEnergy, lastIdx);
    509     y = Interpolation(lastIdx, theEnergy);
    510   }
    511   return y;
    512 }




One more bin edge than value ? Not in the below ? Artificial repetition of last line probably ?
--------------------------------------------------------------------------------------------------

::

    O[blyth@localhost junotop]$ cat data/Simulation/DetSim/Material/LS/RINDEX
    1.55                *eV   1.4781              
    1.79505             *eV   1.48                
    2.10499             *eV   1.4842              
    2.27077             *eV   1.4861              
    2.55111             *eV   1.4915              
    2.84498             *eV   1.4955              
    3.06361             *eV   1.4988              
    4.13281             *eV   1.5264              
    6.2                 *eV   1.6185              
    6.526               *eV   1.6176              
    6.889               *eV   1.527               
    7.294               *eV   1.5545              
    7.75                *eV   1.793               
    8.267               *eV   1.7826              
    8.857               *eV   1.6642              
    9.538               *eV   1.5545              
    10.33               *eV   1.4536              
    15.5                *eV   1.4536              
    O[blyth@localhost junotop]$ 


* ~/j/issues/material_properties_one_more_edge_than_value.rst 

Most but not all material RINDEX properties end with a duplicated value, looks like artificial duplication
to provide some value for the last edge.





cks : Three way comparison ckcf.py 
-------------------------------------


::

In [3]: a[0]                                                                                                                                                                                         
Out[3]: 
array([[  8.7623, 141.5149,   1.6834,   0.891 ],
       [  0.206 ,   0.    ,   0.    ,   1.5   ],
       [452.2491, 141.5149,   0.517 ,   0.157 ],
       [  0.    ,   0.    ,   0.    ,   0.    ]], dtype=float32)



    In [8]: b[0]                                                                                                                                                                                         
    Out[8]: 
    array([[  8.7623, 141.4969,   1.6832,   0.8912],
           [  0.2058,   0.    ,   0.    ,   1.5   ]])

    In [9]: c[0]                                                                                                                                                                                         
    Out[9]: 
    array([[  8.7623, 141.5149,   1.6832,   0.8912],
           [  0.2058,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ]])

    In [10]: b.shape                                                                                                                                                                                     
    Out[10]: (10000, 2, 4)

    In [11]: c.shape                                                                                                                                                                                     
    Out[11]: (10000, 4, 4)

    In [12]: b[10]                                                                                                                                                                                       
    Out[12]: 
    array([[  6.6084, 187.6163,   1.597 ,   0.9392],
           [  0.1178,   0.    ,   0.    ,   1.5   ]])

    In [13]: c[10]                                                                                                                                                                                       
    Out[13]: 
    array([[  6.6084, 187.6402,   1.597 ,   0.9392],
           [  0.1178,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ]])

    In [14]: b[100]                                                                                                                                                                                      
    Out[14]: 
    array([[  8.8041, 140.825 ,   1.6748,   0.8956],
           [  0.1979,   0.    ,   0.    ,   1.5   ]])

    In [15]: c[100]                                                                                                                                                                                      
    Out[15]: 
    array([[  8.8041, 140.8429,   1.6748,   0.8956],
           [  0.1979,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ]])

    In [16]: b[1000]                                                                                                                                                                                     
    Out[16]: 
    array([[  7.9196, 156.5544,   1.7896,   0.8382],
           [  0.2975,   0.    ,   0.    ,   1.5   ]])

    In [17]: c[1000]                                                                                                                                                                                     
    Out[17]: 
    array([[  7.9196, 156.5743,   1.7896,   0.8382],
           [  0.2975,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ]])


    In [4]: a[-1]                                                                                                                                                                                        
    Out[4]: 
    array([[  7.4727, 165.9384,   1.6475,   0.9105],
           [  0.171 ,   0.    ,   0.    ,   1.5   ],
           [385.6852, 165.9384,   0.4246,   0.4489],
           [  0.    ,   0.    ,   0.    ,   0.    ]], dtype=float32)

    In [18]: b[-1]                                                                                                                                                                                       
    Out[18]: 
    array([[  7.4727, 165.9173,   1.6479,   0.9102],
           [  0.1715,   0.    ,   0.    ,   1.5   ]])

    In [19]: c[-1]                                                                                                                                                                                       
    Out[19]: 
    array([[  7.4727, 165.9384,   1.6479,   0.9102],
           [  0.1715,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ],
           [  0.    ,   0.    ,   0.    ,   0.    ]])

    In [20]:                                                   





python energy very closely matches the G4Cerenkov_modified
------------------------------------------------------------

::


    In [15]: c[:,0,0]                                                                                                                                                                                    
    Out[15]: array([8.7623, 6.2018, 5.1259, ..., 4.111 , 7.8475, 7.4727])

    In [16]: b[:,0,0]                                                                                                                                                                                    
    Out[16]: array([8.7623, 6.2018, 5.1259, ..., 4.111 , 7.8475, 7.4727])

    In [17]: bc = b[:,0,0] - c[:,0,0]                                                                                                                                                                    

    In [18]: bc.min()                                                                                                                                                                                    
    Out[18]: -1.7763568394002505e-15

    In [19]: bc.max()                                                                                                                                                                                    
    Out[19]: 1.7763568394002505e-15






8/10k are way off::


    In [20]: np.histogram( a[:,0,0] - b[:,0,0], 100 )                                                                                                                                                    
    Out[20]: 
    (array([   1,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 9992,    0,
               0,    0,    0,    1,    0,    0,    0,    0,    0,    1,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    1]),
     array([-2.8501, -2.8042, -2.7584, -2.7125, -2.6667, -2.6208, -2.575 , -2.5291, -2.4833, -2.4374, -2.3916, -2.3457, -2.2999, -2.254 , -2.2082, -2.1623, -2.1165, -2.0707, -2.0248, -1.979 , -1.9331,
            -1.8873, -1.8414, -1.7956, -1.7497, -1.7039, -1.658 , -1.6122, -1.5663, -1.5205, -1.4746, -1.4288, -1.3829, -1.3371, -1.2912, -1.2454, -1.1995, -1.1537, -1.1078, -1.062 , -1.0161, -0.9703,
            -0.9244, -0.8786, -0.8327, -0.7869, -0.741 , -0.6952, -0.6493, -0.6035, -0.5576, -0.5118, -0.466 , -0.4201, -0.3743, -0.3284, -0.2826, -0.2367, -0.1909, -0.145 , -0.0992, -0.0533, -0.0075,
             0.0384,  0.0842,  0.1301,  0.1759,  0.2218,  0.2676,  0.3135,  0.3593,  0.4052,  0.451 ,  0.4969,  0.5427,  0.5886,  0.6344,  0.6803,  0.7261,  0.772 ,  0.8178,  0.8637,  0.9095,  0.9554,
             1.0012,  1.047 ,  1.0929,  1.1387,  1.1846,  1.2304,  1.2763,  1.3221,  1.368 ,  1.4138,  1.4597,  1.5055,  1.5514,  1.5972,  1.6431,  1.6889,  1.7348]))


    In [21]: deviants = np.abs( a[:,0,0] - b[:,0,0] ) > 0.001                                                                                                                                            

    In [22]: a[deviants]                                                                                                                                                                                 
    Out[22]: 
    array([[[  4.3884, 282.5663,   1.5378,   0.9754],
            [  0.0485,   0.    ,   0.    ,   1.5   ],
            [226.4955, 282.5663,   0.2035,   0.0268],
            [  0.    ,   0.    ,   0.    ,   0.    ]],

           [[  6.0812, 203.9058,   1.6132,   0.9298],
            [  0.1354,   0.    ,   0.    ,   1.5   ],
            [313.8704, 203.9058,   0.3248,   0.1889],
            [  0.    ,   0.    ,   0.    ,   0.    ]],

           [[  8.3739, 148.0784,   1.7613,   0.8516],
            [  0.2747,   0.    ,   0.    ,   1.5   ],
            [432.2036, 148.0784,   0.4892,   0.3752],
            [  0.    ,   0.    ,   0.    ,   0.    ]],

           [[  7.89  , 157.1604,   1.7902,   0.8379],
            [  0.2979,   0.    ,   0.    ,   1.5   ],
            [407.2274, 157.1604,   0.4545,   0.2969],
            [  0.    ,   0.    ,   0.    ,   0.    ]],

           [[  6.7663, 183.2618,   1.5578,   0.9629],
            [  0.0729,   0.    ,   0.    ,   1.5   ],
            [349.2272, 183.2618,   0.3739,   0.2424],
            [  0.    ,   0.    ,   0.    ,   0.    ]],

           [[  9.0393, 137.1786,   1.635 ,   0.9174],
            [  0.1583,   0.    ,   0.    ,   1.5   ],
            [466.545 , 137.1786,   0.5369,   0.5272],
            [  0.    ,   0.    ,   0.    ,   0.    ]],

           [[  5.117 , 242.3309,   1.5702,   0.9553],
            [  0.0874,   0.    ,   0.    ,   1.5   ],
            [264.1017, 242.3309,   0.2557,   0.0638],
            [  0.    ,   0.    ,   0.    ,   0.    ]],

           [[  8.4108, 147.4297,   1.7539,   0.8552],
            [  0.2686,   0.    ,   0.    ,   1.5   ],
            [434.1053, 147.4297,   0.4918,   0.8946],
            [  0.    ,   0.    ,   0.    ,   0.    ]]], dtype=float32)

    In [23]: b[deviants]                                                                                                                                                                                 
    Out[23]: 
    array([[[  7.2384, 171.2861,   1.5507,   0.9673],
            [  0.0644,   0.    ,   0.    ,   1.5   ]],

           [[  4.3465, 285.253 ,   1.5359,   0.9766],
            [  0.0462,   0.    ,   0.    ,   1.5   ]],

           [[  7.1317, 173.8487,   1.5435,   0.9718],
            [  0.0555,   0.    ,   0.    ,   1.5   ]],

           [[  7.3915, 167.7392,   1.6055,   0.9343],
            [  0.1271,   0.    ,   0.    ,   1.5   ]],

           [[  6.0449, 205.1064,   1.6116,   0.9308],
            [  0.1337,   0.    ,   0.    ,   1.5   ]],

           [[  8.7732, 141.3213,   1.681 ,   0.8923],
            [  0.2038,   0.    ,   0.    ,   1.5   ]],

           [[  7.647 , 162.1349,   1.7391,   0.8625],
            [  0.2561,   0.    ,   0.    ,   1.5   ]],

           [[  9.4831, 130.7423,   1.5633,   0.9595],
            [  0.0794,   0.    ,   0.    ,   1.5   ]]])

    In [24]:                          



Excluding the 8 deviants gives a very close energy match::


    In [27]: aa = a[np.logical_not(deviants)]                                                                                                                                                            

    In [28]: bb = b[np.logical_not(deviants)]                                                                                                                                                            

    In [29]: aa[:,0,0] - bb[:,0,0]                                                                                                                                                                       
    Out[29]: array([ 0., -0., -0., ...,  0.,  0.,  0.])

    In [30]: ab = aa[:,0,0] - bb[:,0,0]                                                                                                                                                                  

    In [31]: ab.min()                                                                                                                                                                                    
    Out[31]: -1.8984079375172769e-06

    In [32]: ab.max()                                                                                                                                                                                    
    Out[32]: 2.041459083557129e-06


    In [33]: deviants                                                                                                                                                                                    
    Out[33]: array([False, False, False, ..., False, False, False])

    In [34]: np.where(deviants)                                                                                                                                                                          
    Out[34]: (array([ 213,  817, 1351, 1902, 2236, 3114, 4812, 6139]),)




Looking at wavelength_cfplot curious that discrepancy peaks at 330nm and just prior::

    ARG=5 ipython -i wavelength_cfplot.py

What is special about there ? The rindex is close to BetaIndex in that range. So are near the threshold ?
Near threshold presumably means more samples are rejected to find a permissable energy so higher probability 
for difference from close to cuts ?


Look at rejection looping:: 

    In [60]: np.where( a_loop != b_loop )                                                                                                                                                                
    Out[60]: (array([ 213,  817, 1351, 1902, 2236, 3114, 4812, 6139]),)


    In [63]: b_loop.min(), b_loop.max()                                                                                                                                                                  
    Out[63]: (1, 42)

    In [64]: c_loop.min(), c_loop.max()                                                                                                                                                                  
    Out[64]: (1, 42)

    In [65]: a_loop.min(), a_loop.max()                                                                                                                                                                  
    Out[65]: (1, 42)



::

    In [2]: np.unique( a_loop , return_counts=True )                                                                                                                                                     
    Out[2]: 
    (array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 40, 42], dtype=int32),
     array([1912, 1521, 1204,  997,  822,  694,  540,  459,  348,  309,  232,  181,  146,  110,   87,   84,   57,   52,   51,   39,   33,   22,   18,   17,   15,   11,    8,    8,    3,    1,    2,    4,
               1,    5,    2,    1,    1,    2,    1]))

    In [3]: np.unique( b_loop , return_counts=True )                                                                                                                                                     
    Out[3]: 
    (array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 40, 42], dtype=int32),
     array([1913, 1521, 1203,  999,  821,  694,  538,  459,  348,  309,  233,  181,  145,  110,   87,   84,   57,   52,   53,   40,   32,   22,   17,   17,   15,   11,    8,    8,    3,    1,    2,    4,
               1,    5,    2,    1,    1,    2,    1]))

    In [4]: np.unique( c_loop , return_counts=True )                                                                                                                                                     
    Out[4]: 
    (array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 40, 42]),
     array([1913, 1521, 1203,  999,  821,  694,  538,  459,  348,  309,  233,  181,  145,  110,   87,   84,   57,   52,   53,   40,   32,   22,   17,   17,   15,   11,    8,    8,    3,    1,    2,    4,
               1,    5,    2,    1,    1,    2,    1]))



Using hc_eVnm/15.5 hc_eVnm/1.55 rather than the close 80. 800. gets the wavelengths very close::


    In [1]: a[:,0,1]                                                                                                                                                                                     
    Out[1]: array([141.4969, 199.9157, 241.8792, ..., 301.5883, 157.9929, 165.9173], dtype=float32)

    In [2]: b[:,0,1]                                                                                                                                                                                     
    Out[2]: array([141.4969, 199.9157, 241.8792, ..., 301.5883, 157.9929, 165.9173])

    In [3]: c[:,0,1]                                                                                                                                                                                     
    Out[3]: array([141.4969, 199.9157, 241.8792, ..., 301.5883, 157.9929, 165.9173])


Huh ... wow. Getting precisely the same range gets aligned 10k running to match precisely.
Looks like perfect match in aligned 10k running, "chi2" zero.

BUT : comparing non-aligned 1M samples in wavelength_cfplot.py still get bad chi2:: 

    ARG=6 ipython -i wavelength_cfplot.py

* so the problem is finding the cause of extreme fragility in some regions 

TODO: cook (3M, 16,16) randoms in qctx and check using that 


