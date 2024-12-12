RNG_flexibility
===================


DONE : removed old s_mock_curand.h 
----------------------------------------

::

    typedef srngcpu curandStateXORWOW ; 
    typedef srngcpu curandState_t ; 
    //
    inline float curand_uniform(curandState_t* state ){         return state->generate_float() ; }
    inline double curand_uniform_double(curandState_t* state ){ return state->generate_double() ; }



::

    P[blyth@localhost opticks]$ opticks-f s_mock_curand.h
    ./qudarap/tests/QSim_MockTest.cc:    sysrap/s_mock_curand.h 
    ./qudarap/tests/QSim_MockTest.cc:#include "scurand.h"    // includes s_mock_curand.h when MOCK_CURAND OR MOCK_CUDA defined 
    ./sysrap/CMakeLists.txt:    s_mock_curand.h
    ./sysrap/s_mock_curand.h:s_mock_curand.h : AIMING TO REMOVE  
    P[blyth@localhost opticks]$ 



DONE : Replaced s_mock_curand.h with more explicit + easier to use approach
-------------------------------------------------------------------------------

Prefer more explicit approach, namely:

1. all use of curand that is not specific to one state type 
   should use RNG for the type

2. usage later specialized to specific type with eg::

    // on CPU
    #include "srngcpu.h"
    using RNG = srngcpu ;

    // on GPU 
    #include "curand_kernel.h"
    using RNG = curandStateXORWOW ;




Want to be able to switch the RNG impl with a recompile to compare different impl
------------------------------------------------------------------------------------

    
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



curand XORWOW vs Philox4_32_10
-----------------------------------

* https://forums.developer.nvidia.com/t/quick-benchmark-comparison-of-different-parallel-random-number-generators/40591

* https://forums.developer.nvidia.com/t/should-a-kernel-initializing-random-states-with-curand-init-be-so-slow/62666


Getting things to work with Philox
--------------------------------------

::

    FAILS:  2   / 215   :  Thu Dec 12 22:35:42 2024   
      6  /21  Test #6  : QUDARapTest.QSimTest                          ***Failed                      5.34   
      13 /21  Test #13 : QUDARapTest.QSimWithEventTest                 ***Failed                      4.94   


  
Checking curand_init and curand_uniform for different curandState impl
--------------------------------------------------------------------------

Takeaway:

* Philox4_32_10 is winner 
* Philox4_32_10_OpticksLite comes close, but not worth hassle if does not signifcantly improve 
* CAVEAT : this is just doing curand_init and curand_uniform : the stack that required will
  implications on the rest of the simulation  

  * so that means need to make a similar comparison for full simulation 


::
    
    P[blyth@localhost opticks]$ ~/o/sysrap/tests/curand_uniform_test.sh
             BASH_SOURCE : /home/blyth/o/sysrap/tests/curand_uniform_test.sh 
                    name : curand_uniform_test 
                     src : curand_uniform_test.cu 
                  script : curand_uniform_test.py 
                     bin : /data/blyth/opticks/curand_uniform_test/curand_uniform_test 
                    FOLD : /data/blyth/opticks/curand_uniform_test 
                     OPT : -use_fast_math -DWITH_CURANDLITE 
                      NI : 10000000 
                      NJ : 16 
     t1 - t0 : output allocations [us] 1883714
     rngmax 10000000 rngmax/M 10 available_chunk 24 all.num/M 200 rngmax/M 10 d0 0x7f026a000000
     i   0 ck.ref.num/M    1 count/M    0 remaining/M   10 partial_read NO  num/M    1 d 0x7f026a000000
     i   1 ck.ref.num/M    1 count/M    1 remaining/M    9 partial_read NO  num/M    1 d 0x7f026cdc6c00
     i   2 ck.ref.num/M    1 count/M    2 remaining/M    8 partial_read NO  num/M    1 d 0x7f026fb8d800
     i   3 ck.ref.num/M    1 count/M    3 remaining/M    7 partial_read NO  num/M    1 d 0x7f0272954400
     i   4 ck.ref.num/M    1 count/M    4 remaining/M    6 partial_read NO  num/M    1 d 0x7f027571b000
     i   5 ck.ref.num/M    1 count/M    5 remaining/M    5 partial_read NO  num/M    1 d 0x7f02784e1c00
     i   6 ck.ref.num/M    1 count/M    6 remaining/M    4 partial_read NO  num/M    1 d 0x7f027b2a8800
     i   7 ck.ref.num/M    1 count/M    7 remaining/M    3 partial_read NO  num/M    1 d 0x7f027e06f400
     i   8 ck.ref.num/M    1 count/M    8 remaining/M    2 partial_read NO  num/M    1 d 0x7f0280e36000
     i   9 ck.ref.num/M    1 count/M    9 remaining/M    1 partial_read NO  num/M    1 d 0x7f0283bfcc00
    SCurandState::loadAndUpload complete YES rngmax/M 10 rngmax 10000000 digest ffe00cfef9d97aeef4c1cf085fd46a6a(cf md5sum of cat-ed chunk(s))
     t2 - t1 : loadAndUpload [us] 3273220

     dt0 3273228 ms 119.979263 [t1-t0;us]   120047 states NO  download NO  four_by_four NO  name XORWOW
     dt0 3393335 ms   7.869184 [t1-t0;us]     7877 states YES download NO  four_by_four NO  name XORWOW
     dt0 3401224 ms   5.373152 [t1-t0;us]     5380 states NO  download NO  four_by_four NO  name Philox4_32_10
     dt0 3406620 ms   7.155616 [t1-t0;us]     7163 states NO  download NO  four_by_four NO  name Philox4_32_10_OpticksLite

     dt0 3413794 ms 117.149826 [t1-t0;us]   117157 states NO  download NO  four_by_four YES name XORWOW
     dt0 3530967 ms   7.495072 [t1-t0;us]     7502 states YES download NO  four_by_four YES name XORWOW
     dt0 3538480 ms   3.867456 [t1-t0;us]     3875 states NO  download NO  four_by_four YES name Philox4_32_10
     dt0 3542370 ms   3.880960 [t1-t0;us]     3890 states NO  download NO  four_by_four YES name Philox4_32_10_OpticksLite

     dt0 3546271 ms  85.575714 [t1-t0;us]    86531 states NO  download NO  four_by_four NO  name XORWOW
     dt0 3632817 ms   8.580960 [t1-t0;us]     8587 states YES download NO  four_by_four NO  name XORWOW
     dt0 3641415 ms   3.845760 [t1-t0;us]     3857 states NO  download NO  four_by_four NO  name Philox4_32_10
     dt0 3645282 ms   3.860480 [t1-t0;us]     3868 states NO  download NO  four_by_four NO  name Philox4_32_10_OpticksLite

     dt0 3649160 ms  84.454819 [t1-t0;us]    85468 states NO  download NO  four_by_four YES name XORWOW
     dt0 3734642 ms   7.616608 [t1-t0;us]     7623 states YES download NO  four_by_four YES name XORWOW
     dt0 3742276 ms   3.861952 [t1-t0;us]     3872 states NO  download NO  four_by_four YES name Philox4_32_10
     dt0 3746159 ms   3.862528 [t1-t0;us]     3869 states NO  download NO  four_by_four YES name Philox4_32_10_OpticksLite
    f

    CMDLINE:curand_uniform_test.py
    f.base:/data/blyth/opticks/curand_uniform_test

      : f.RNG4                                             :       (10000000, 16) : 0:08:43.330069 
      : f.RNG5                                             :       (10000000, 16) : 0:08:39.881046 
      : f.RNG6                                             :       (10000000, 16) : 0:08:36.458023 
      : f.RNG7                                             :       (10000000, 16) : 0:08:33.058000 

     min_stamp : 2024-12-12 16:31:01.412687 
     max_stamp : 2024-12-12 16:31:11.684756 
     dif_stamp : 0:00:10.272069 
     age_stamp : 0:08:33.058000 



qrng.h how to do the curand_init there ?
--------------------------------------------

::

    1012 QUALIFIERS void curand_init(unsigned long long seed,
    1013                                  unsigned long long subsequence,
    1014                                  unsigned long long offset,
    1015                                  curandStatePhilox4_32_10_t *state)
    1016 {
    1017     state->ctr = make_uint4(0, 0, 0, 0);
    1018     state->key.x = (unsigned int)seed;
    1019     state->key.y = (unsigned int)(seed>>32);
    1020     state->STATE = 0;
    1021     state->boxmuller_flag = 0;
    1022     state->boxmuller_flag_double = 0;
    1023     state->boxmuller_extra = 0.f;
    1024     state->boxmuller_extra_double = 0.;
    1025     skipahead_sequence(subsequence, state);
    1026     skipahead(offset, state);
    1027 }



skipahead:offset
   ctr.xyzw

skipahead_sequence:subsequence  
   ctr.zw



::

    106 QUALIFIERS void Philox_State_Incr(curandStatePhilox4_32_10_t* s, unsigned long long n)
    107 {
    108    unsigned int nlo = (unsigned int)(n);
    109    unsigned int nhi = (unsigned int)(n>>32);
    110 
    111    s->ctr.x += nlo;
    112    if( s->ctr.x < nlo )
    113       nhi++;
    114 
    115    s->ctr.y += nhi;
    116    if(nhi <= s->ctr.y)
    117       return;
    118    if(++s->ctr.z) return;
    119    ++s->ctr.w;
    120 }
    121 
    122 QUALIFIERS void Philox_State_Incr_hi(curandStatePhilox4_32_10_t* s, unsigned long long n)
    123 {
    124    unsigned int nlo = (unsigned int)(n);
    125    unsigned int nhi = (unsigned int)(n>>32);
    126 
    127    s->ctr.z += nlo;
    128    if( s->ctr.z < nlo )
    129       nhi++;
    130 
    131    s->ctr.w += nhi;
    132 }

     985 QUALIFIERS void skipahead_sequence(unsigned long long n, curandStatePhilox4_32_10_t *state)
     986 {
     987     Philox_State_Incr_hi(state, n);
     988     state->output = curand_Philox4x32_10(state->ctr,state->key);
     989 }

     961 QUALIFIERS void skipahead(unsigned long long n, curandStatePhilox4_32_10_t *state)
     962 {
     963     state->STATE += (n & 3);
     964     n /= 4;
     965     if( state->STATE > 3 ){
     966         n += 1;
     967         state->STATE -= 4;
     968     }
     969     Philox_State_Incr(state, n);
     970     state->output = curand_Philox4x32_10(state->ctr,state->key);
     971 }





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

* nope went for the same simple approach::

      #include "srngcpu.h"
      using RNG = srngcpu ; 



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


::

    P[blyth@localhost tests]$ grep curandState *.*
    curand_uniform_test.cu:#include "curandlite/curandStatePhilox4_32_10_OpticksLite.h"
    curand_uniform_test.cu:using opticks_curandState_t = curandStatePhilox4_32_10_OpticksLite ; 
    curand_uniform_test.cu:        printf("test_curand_uniform<curandStateXORWOW>()"); 
    curand_uniform_test.cu:        test_curand_uniform<curandStateXORWOW>();
    curand_uniform_test.cu:        printf("test_curand_uniform<curandStatePhilox4_32_10>()"); 
    curand_uniform_test.cu:        test_curand_uniform<curandStatePhilox4_32_10>();
    curand_uniform_test.cu:        printf("test_curand_uniform<curandStatePhilox4_32_10_OpticksLite>()"); 
    curand_uniform_test.cu:        test_curand_uniform<curandStatePhilox4_32_10_OpticksLite>();
    curand_uniform_test.cu:        printf("test_curand_uniform<opticks_curandState_t>()"); 
    curand_uniform_test.cu:        test_curand_uniform<opticks_curandState_t>();
    scerenkov_test.cc:    curandStateXORWOW rng(1u); 
    SCurandState_test.cc:    implement loading of any number of curandState within the range 
    scurand_test.cc:    curandStateXORWOW rng(1u) ;   
    s_mock_curand_test.cc:void test_mock_curand_0(curandState_t& rng)
    s_mock_curand_test.cc:void test_mock_curand_1(curandStateXORWOW& rng)
    s_mock_curand_test.cc:    curandState_t rng(1u) ;   
    s_mock_curand_test.cc:    curandStateXORWOW rng(1u) ;   
    stmm_vs_sboundary_test.cc:    curandStateXORWOW rng(1u) ; 
    storch_test.cc:    curandStateXORWOW rng(1u); 
    P[blyth@localhost tests]$ 



Maybe eliminate scurand.h use from qudarap ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nope its used from some test code in qcerenkov.h : should split that off perhaps
for kernel cleanup. 


srng.h is misleadingly named, rename to srngcpu.h
--------------------------------------------------

::

    P[blyth@localhost qudarap]$ opticks-f srng.h 
    ./sysrap/s_mock_curand.h:#include "srng.h"
    ./sysrap/scarrier.h:#include "srng.h"
    ./sysrap/tests/srng_test.cc:#include "srng.h"
    ./sysrap/scurand.h:   #include "srng.h"
    P[blyth@localhost opticks]$ 



