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


* 

* potential cause : float vs double, if so need to drill down as to exactly where
 


Getting G4Cerenkov_modified to use precooked randoms using OpticksRandom
-----------------------------------------------------------------------------

::

    G4Cerenkov_modifiedTest::PSDI [BetaInverse_1.500_step_length_100.000_SKIP_CONTINUE]
     i      0 rand0    0.74022 rand1    0.43845
     i      0 rand0    0.51701 rand1    0.15699

    G4Cerenkov_modified


    In [26]: np.set_printoptions(precision=6)

    In [27]: a[0,0,:4]
    Out[27]: array([0.740219, 0.438451, 0.517013, 0.156989])




