RNG_flexibility
===================

Want to be able to switch the RNG impl with a recompile to compare:
    
+---------------------------------------+----------------+--------------------------------------------------+
|                                       |  sizeof bytes  |   notes                                          |
+=======================================+================+==================================================+
| curandStateXORWOW                     |    48          |  curand default, expensive init => complications |
+---------------------------------------+----------------+--------------------------------------------------+
| curandStatePhilox4_32_10              |    64          |  cheap init (TODO: check in practice)            |
+---------------------------------------+----------------+--------------------------------------------------+
| curandStatePhilox4_32_10_OpticksLite  |    32          |  slim state to uint4 + uint2, gets padded to 32  |
+---------------------------------------+----------------+--------------------------------------------------+

/usr/local/cuda/include/curand_philox4x32_x.h::

    092 struct curandStatePhilox4_32_10 {
     93    uint4 ctr;
     94    uint4 output;
     95    uint2 key;
     96    unsigned int STATE;

     97    int boxmuller_flag;
     98    int boxmuller_flag_double;
     99    float boxmuller_extra;
    100    double boxmuller_extra_double;
    101 };

~/opticks/sysrap/curandlite/curandStatePhilox4_32_10_OpticksLite.h::

     42 struct curandStatePhilox4_32_10_OpticksLite
     43 {
     44     uint4 ctr ;
     45     uint2 key ;
     46     // looks like 6*4=24 bytes, but gets padded to 32 bytes
     47 };

/usr/local/cuda/include/curand_kernel::

     140 struct curandStateXORWOW {
     141     unsigned int d, v[5];

     142     int boxmuller_flag;
     143     int boxmuller_flag_double;
     144     float boxmuller_extra;
     145     double boxmuller_extra_double;
     146 };


qudarap code flexibility
---------------------------

qrng.h::
     
    #if defined(MOCK_CUDA)
    #else
    struct curandStateXORWOW ; 
    using RNG = curandStateXORWOW ; 
    #endif


Then changing all curandState curandStateXORWOW to RNG in qudarap, worked ok.


sysrap ? 
-----------

Not so easy in sysrap, due to mock cuda complications with scurand.h and scarrier.h
The problem being they need to work with both with mock and real cuda ? 

Maybe templated generate method etc can avoid the complication ? 


::

    P[blyth@localhost sysrap]$ opticks-f scurand.h 
    ./qudarap/qcerenkov.h:#include "scurand.h"
    ./qudarap/tests/QSim_MockTest.cc:    sysrap/scurand.h 
    ./qudarap/tests/QSim_MockTest.cc:#include "scurand.h"    // includes s_mock_curand.h when MOCK_CURAND OR MOCK_CUDA defined 
    ./qudarap/QSim.cu:#include "scurand.h"
    ./qudarap/qsim.h:#include "scurand.h"
    ./qudarap/QRng.cu:#include "scurand.h"
    ./sysrap/CMakeLists.txt:    scurand.h  
    ./sysrap/SGenerate.h:#include "scurand.h"   // without MOCK_CURAND this is an empty struct only 
    ./sysrap/s_mock_curand.h:This is conditionally included by scurand.h 
    ./sysrap/scerenkov.h:#include "scurand.h"
    ./sysrap/sboundary.h:#include "scurand.h"
    ./sysrap/sscint.h:#include "scurand.h"
    ./sysrap/storch.h:#include "scurand.h"
    ./sysrap/tests/sboundary_test.cc:#include "scurand.h"
    ./sysrap/tests/scarrier_test.cc:#include "scurand.h"    // this brings in s_mock_curand.h for CPU when MOCK_CURAND macro is defined 
    ./sysrap/tests/scerenkov_test.cc:#include "scurand.h"    // this brings in s_mock_curand.h for CPU when MOCK_CURAND macro is defined 
    ./sysrap/tests/scurand_test.cc:#include "scurand.h"
    ./sysrap/tests/stmm_vs_sboundary_test.cc:#include "scurand.h"
    ./sysrap/tests/storch_test.cc:#include "scurand.h"    // this brings in s_mock_curand.h for CPU when MOCK_CURAND macro is defined 
    P[blyth@localhost opticks]$ 



    P[blyth@localhost sysrap]$ opticks-f scarrier.h 
    ./qudarap/qsim.h:#include "scarrier.h"
    ./sysrap/CMakeLists.txt:    scarrier.h
    ./sysrap/SEvent.cc:#include "scarrier.h"
    ./sysrap/SGenerate.h:#include "scarrier.h"
    ./sysrap/tests/scarrier_test.cc:scarrier_test.cc : CPU tests of scarrier.h CUDA code using mocking 
    ./sysrap/tests/scarrier_test.cc:#include "scarrier.h"
    P[blyth@localhost opticks]$ 



srng.h is misleadingly named, rename to srngcpu.h
--------------------------------------------------

::

    P[blyth@localhost qudarap]$ opticks-f srng.h 
    ./sysrap/s_mock_curand.h:#include "srng.h"
    ./sysrap/scarrier.h:#include "srng.h"
    ./sysrap/tests/srng_test.cc:#include "srng.h"
    ./sysrap/scurand.h:   #include "srng.h"
    P[blyth@localhost opticks]$ 



