cuda-11_2-optix-6_5-double-precision-symbols-compilation-fail
================================================================


Two Test Fails with OptiX 6.5 CUDA 11.2 Driver 460
-----------------------------------------------------

::

    Thanks for your last reply. I attempted to compiled Opticks. 

    We use OptiX 6.5 , cuda-11.2, NVIDIA GPU Driver 460.32. And then I follow
    your instruction from   this website
    https://simoncblyth.bitbucket.io/opticks/docs/install.html#version-requirements.
    Finally, the Opticks has been compiled successfully with the help of Tao. In
    the whole process, we meet the following error

    1.    I test the  opticks-t bash function, but unfortunately, 2 test failed. the out put information shows like that:


         SLOW: tests taking longer that 15 seconds
      8  /38  Test #8  : CFG4Test.CG4Test                              Passed                         19.72  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Passed                         23.90  


    FAILS:  2   / 453   :  Fri Mar 19 16:35:32 2021   
      5  /32  Test #5  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     0.26   
      29 /32  Test #29 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     0.27 
     


Double Precision Symbols PTX Compilation Fail
----------------------------------------------------

::


    [ihep@localhost ~]$ Roots3And4Test    
    2021-03-19 18:48:10.448 FATAL [154846] [OptiXTest::OptiXTest@89] /home/ihep/local/opticks/installcache/PTX//OptiXRap_generated_Roots3And4Test.cu.ptx
    2021-03-19 18:48:10.449 INFO  [154846] [OptiXTest::init@95] OptiXTest::init cu Roots3And4Test.cu ptxpath /home/ihep/local/opticks/installcache/PTX//OptiXRap_generated_Roots3And4Test.cu.ptx raygen Roots3And4Test exception exception
    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtProgramCreateFromPTXFile(RTcontext, const char*, const char*, RTprogram_api**)" caught exception: 
      Compile Error: 
            Unknown usage of symbol: __cudart_i2opi_d 
      at: 
            [ Instruction:   %14 = extractelement <2 x i32> %13, i32 0, !dbg !11, contained in basic block: LBB2_3, 
            in function: __internal_trig_reduction_slowpathd, 
            in module: Canonical__Z14Roots3And4Testv 
            from /home/ihep/local/opticks/installcache/PTX//OptiXRap_generated_Roots3And4Test.cu.ptx 
            associated DI info (Dir/File/Line): OPTIX/generated/ generated 2811, LL file line and column: 2811:1 ])
    Aborted (core dumped)



     [ihep@localhost tests]$ ./intersectAnalyticTest.sh    
    ##########  intersectAnalyticTest --cu iaTorusTest.cu  ############  

    SAr::dump SAr _argc 3 (  intersectAnalyticTest --cu iaTorusTest.cu ) 
    2021-03-19 18:54:01.326 INFO  [155421] [main@71]  cu_name iaTorusTest.cu progname iaTorusTest
    2021-03-19 18:54:01.389 INFO  [155421] [main@80]  stack_size 2688
    2021-03-19 18:54:01.389 FATAL [155421] [OptiXTest::OptiXTest@89] /home/ihep/local/opticks/installcache/PTX/tests/intersectAnalyticTest_generated_iaTorusTest.cu.ptx
    2021-03-19 18:54:01.389 INFO  [155421] [OptiXTest::init@95] OptiXTest::init cu iaTorusTest.cu ptxpath /home/ihep/local/opticks/installcache/PTX/tests/intersectAnalyticTest_generated_iaTorusTest.cu.ptx raygen iaTorusTest exception exception
    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtProgramCreateFromPTXFile(RTcontext, const char*, const char*, RTprogram_api**)" caught exception: 
      Compile Error: 
          Unknown usage of symbol: __cudart_sin_cos_coeffs 
      at: 
          [ Instruction:   %597 = insertelement <2 x double> %596, double %592, i32 1, !dbg !11, contained in basic block: LBB0_63, 
          in function: _Z11iaTorusTestv_ptx0x9111921bcae80899, 
          in module:   Canonical__Z11iaTorusTestv 
          from /home/ihep/local/opticks/installcache/PTX/tests/intersectAnalyticTest_generated_iaTorusTest.cu.ptx 
          associated DI info (Dir/File/Line): OPTIX/generated/ generated 108, LL file line and column: 108:1 ])

    ./intersectAnalyticTest.sh: line 37: 155421 Aborted                 (core dumped) intersectAnalyticTest --cu iaTorusTest.cu
    === ia-test : ERROR non-zero RC 134




Check Forum
--------------


* https://forums.developer.nvidia.com/search?q=Unknown%20usage%20of%20symbol%20category%3A167
* https://forums.developer.nvidia.com/t/mdl-compiled-ptx-optix-problem/83048

* https://forums.developer.nvidia.com/search?q=__cudart_i2opi_d%20
* https://forums.developer.nvidia.com/t/tesla-k20c-and-error-locating-ptx-symbol-cudart-i2opi-f-1049181-new-optix-3-0/27747/3




* :google:`__cudart_i2opi_f`

Fastmath ptx

* https://github.com/numba/numba/issues/6183



* https://pages.mtu.edu/~struther/Courses/OLD/Other/Older/5903_2010/MathApprox.c

::

    /* 160 bits of 2/PI for Payne-Hanek style argument reduction. */
    static __constant__ unsigned int __cudart_i2opi_f [] = {
      0x3c439041,
      0xdb629599,
      0xf534ddc0,
      0xfc2757d1,
      0x4e441529,
      0xa2f9836e,
    };



* https://stackoverflow.com/questions/30463616/payne-hanek-algorithm-implementation-in-c






* :google:`__cudart_sin_cos_coeffs`


* https://forums.developer.nvidia.com/t/working-optix-3-app-fails-optix-4/57134


* https://jar-download.com/artifacts/org.jcuda/jcuda-vec/0.0.2/source-code/kernels/JCudaVec_kernels_double_32_cc30.ptx


::

    .const .align 8 .b8 __cudart_i2opi_d[144] = {8, 93, 141, 31, 177, 95, 251, 107, 234, 146, 82, 138, 247, 57, 7, 61, 123, 241, 229, 235, 199, 186, 39, 117, 45, 234, 95, 158, 102, 63, 70, 79, 183, 9, 203, 39, 207, 126, 54, 109, 31, 109, 10, 90, 139, 17, 47, 239, 15, 152, 5, 222, 255, 151, 248, 31, 59, 40, 249, 189, 139, 95, 132, 156, 244, 57, 83, 131, 57, 214, 145, 57, 65, 126, 95, 180, 38, 112, 156, 233, 132, 68, 187, 46, 245, 53, 130, 232, 62, 167, 41, 177, 28, 235, 29, 254, 28, 146, 209, 9, 234, 46, 73, 6, 224, 210, 77, 66, 58, 110, 36, 183, 97, 197, 187, 222, 171, 99, 81, 254, 65, 144, 67, 60, 153, 149, 98, 219, 192, 221, 52, 245, 209, 87, 39, 252, 41, 21, 68, 78, 110, 131, 249, 162};
    .const .align 8 .b8 __cudart_sin_cos_coeffs[128] = {186, 94, 120, 249, 101, 219, 229, 61, 70, 210, 176, 44, 241, 229, 90, 190, 146, 227, 172, 105, 227, 29, 199, 62, 161, 98, 219, 25, 160, 1, 42, 191, 24, 8, 17, 17, 17, 17, 129, 63, 84, 85, 85, 85, 85, 85, 197, 191, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 129, 253, 32, 131, 255, 168, 189, 40, 133, 239, 193, 167, 238, 33, 62, 217, 230, 6, 142, 79, 126, 146, 190, 233, 188, 221, 25, 160, 1, 250, 62, 71, 93, 193, 22, 108, 193, 86, 191, 81, 85, 85, 85, 85, 85, 165, 63, 0, 0, 0, 0, 0, 0, 224, 191, 0, 0, 0, 0, 0, 0, 240, 63};




* https://stackoverflow.com/questions/16941047/function-properties-for-internal-trig-reduction-slowpathd

::

    __internal_trig_reduction_slowpathd() is an internal subroutine in the CUDA
    math library. It is used to perform accurate argument reduction for
    double-precision trig functions (sin, cos, sincos, tan) when the argument is
    very large in magnitude. A Payne-Hanek style argument reduction is used for
    these large arguments. For sm_20 and up, this is a called subroutine to
    minimize code size in apps that invoke trig functions frequently. You can see
    the code by looking at the file math_functions_dbl_ptx3.h which is in the CUDA
    include file directory.


* https://forums.developer.nvidia.com/t/a-faster-and-more-accurate-implementation-of-sincosf/44620





