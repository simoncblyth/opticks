OptiX_600_CUDA_10.1_test_fails
=================================



Getting better after adding setRayTypeCount
----------------------------------------------

Some of the fails fixed by::

    [blyth@localhost opticks]$ hg diff optixrap/OptiXTest.cc
    diff -r 396804bcf0a5 optixrap/OptiXTest.cc
    --- a/optixrap/OptiXTest.cc Thu Apr 11 23:47:08 2019 +0800
    +++ b/optixrap/OptiXTest.cc Fri Apr 12 13:50:17 2019 +0800
    @@ -40,6 +40,9 @@
                   << description()
                    ; 
     
    +    unsigned num_ray_types = 1; 
    +    context->setRayTypeCount(num_ray_types);  
    +    // without setRayTypeCount get SEGV at launch in OptiX_600, changed default or stricter ? an assert would have been nice !
         context->setEntryPointCount( 1 );
     
         optix::Program raygenProg    = context->createProgramFromPTXFile(m_ptxpath, m_raygen_name);
    [blyth@localhost opticks]$ 



Now down to 3 modes of failure::


    FAILS:
      12 /19  Test #12 : OptiXRapTest.OOtex0Test                       ***Exception: SegFault         1.18   
      13 /19  Test #13 : OptiXRapTest.OOtexTest                        ***Exception: SegFault         1.16   
             
      2 with SEGV at launch (presumably tex changes again)   

      18 /19  Test #18 : OptiXRapTest.intersect_analytic_test          Child aborted***Exception:     1.17   
      19 /19  Test #19 : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     1.14   

      2 with optix::Exception misaligned address

      15 /19  Test #15 : OptiXRapTest.OEventTest                       Child aborted***Exception:     1.40   
      2  /5   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     3.85      
      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     5.31   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     21.78  

      Four with OBufBase assert
      OBufBase::examineBufferFormat(RTformat): Assertion `element_size_bytes == soa*mul' 

      [blyth@localhost opticks]$ date
      Fri Apr 12 13:52:31 CST 2019
  

OOtex0Test::

    (gdb) bt
    #0  0x00007fffe5b0a387 in ?? () from /lib64/libnvoptix.so.1
    #1  0x00007fffe5e3f5d9 in ?? () from /lib64/libnvoptix.so.1
    #2  0x00007fffe5ad8d0e in ?? () from /lib64/libnvoptix.so.1
    #3  0x00007fffe5ad9551 in ?? () from /lib64/libnvoptix.so.1
    #4  0x00007fffe5ad9ffb in ?? () from /lib64/libnvoptix.so.1
    #5  0x00007fffe5ff9094 in ?? () from /lib64/libnvoptix.so.1
    #6  0x00007fffe5f9d996 in ?? () from /lib64/libnvoptix.so.1
    #7  0x000000000040794b in optix::ContextObj::launch (this=0x745a40, entry_point_index=0, image_width=16, image_height=16) at /usr/local/OptiX_600/include/optixu/optixpp_namespace.h:2901
    #8  0x0000000000406463 in main (argc=1, argv=0x7fffffffdaa8) at /home/blyth/opticks/optixrap/tests/OOtex0Test.cc:102
    (gdb) 

     
OOtexTest::
    (gdb) bt
    #0  0x00007fffdf5e0387 in ?? () from /lib64/libnvoptix.so.1
    #1  0x00007fffdf9155d9 in ?? () from /lib64/libnvoptix.so.1
    #2  0x00007fffdf5aed0e in ?? () from /lib64/libnvoptix.so.1
    #3  0x00007fffdf5af551 in ?? () from /lib64/libnvoptix.so.1
    #4  0x00007fffdf5afffb in ?? () from /lib64/libnvoptix.so.1
    #5  0x00007fffdfacf094 in ?? () from /lib64/libnvoptix.so.1
    #6  0x00007fffdfa73996 in ?? () from /lib64/libnvoptix.so.1
    #7  0x000000000040771d in optix::ContextObj::launch (this=0x74b6e0, entry_point_index=0, image_width=16, image_height=16) at /usr/local/OptiX_600/include/optixu/optixpp_namespace.h:2901
    #8  0x000000000040631b in main (argc=1, argv=0x7fffffffdaa8) at /home/blyth/opticks/optixrap/tests/OOtexTest.cc:94
    (gdb) 


intersect_analytic_test::

    // pid 0 
    // csg_intersect_torus_test  r R rmax (10 100 110) ray_origin (-0.646 0.005311 3.947) ray_direction (0.00059 0.0007738 -0.009953) 
    // csg_intersect_torus R r unit (99.9955 9.99955 0.0100005)  oxyz (-64.5971 0.531076 394.682) sxyz (0.0589973 0.0773765 -0.995255 ) t_min (0)   
    // csg_intersect_torus HGIJKL (-301570 378.678 1.66907e+08 1 -793.158 169846)  ABCDE (1 -1586.32 968414 -2.69128e+08 2.86808e+10 ) 
    // csg_intersect_torus qn (-1586.32 968414 -2.69128e+08 2.86808e+10) reverse 0 
    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuEventSynchronize( m_event ) returned (716): Misaligned address)
    Aborted (core dumped)

Roots3And4Test::

    [blyth@localhost okop]$ Roots3And4Test
    2019-04-12 14:14:34.014 INFO  [124780] [OptiXTest::init@39] OptiXTest::init cu Roots3And4Test.cu ptxpath /home/blyth/local/opticks/build/optixrap/OptiXRap_generated_Roots3And4Test.cu.ptx raygen Roots3And4Test exception exception
    2019-04-12 14:14:34.016 INFO  [124780] [OptiXTest::Summary@75] Roots3And4Test cu Roots3And4Test.cu ptxpath /home/blyth/local/opticks/build/optixrap/OptiXRap_generated_Roots3And4Test.cu.ptx raygen Roots3And4Test exception exception
    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuEventSynchronize( m_event ) returned (716): Misaligned address)
    Aborted (core dumped)





Titan RTX
----------

::

    FAILS:
      4  /18  Test #4  : OptiXRapTest.OOMinimalTest                    ***Exception: SegFault         1.55   
      5  /18  Test #5  : OptiXRapTest.OOMinimalRedirectTest            ***Exception: SegFault         1.14   
      11 /18  Test #11 : OptiXRapTest.OOtex0Test                       ***Exception: SegFault         1.58   
      12 /18  Test #12 : OptiXRapTest.OOtexTest                        ***Exception: SegFault         1.53   
      17 /18  Test #17 : OptiXRapTest.intersect_analytic_test          ***Exception: SegFault         2.04   
      18 /18  Test #18 : OptiXRapTest.Roots3And4Test                   ***Exception: SegFault         1.66   

      14 /18  Test #14 : OptiXRapTest.OEventTest                       Child aborted***Exception:     1.37   
      2  /5   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     4.63   
      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     6.38   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     21.63  
    [blyth@localhost opticks]$ 


Titan V
---------

::

    FAILS:
      4  /18  Test #4  : OptiXRapTest.OOMinimalTest                    ***Exception: SegFault         1.64   
      5  /18  Test #5  : OptiXRapTest.OOMinimalRedirectTest            ***Exception: SegFault         1.24   
      11 /18  Test #11 : OptiXRapTest.OOtex0Test                       ***Exception: SegFault         1.62   
      12 /18  Test #12 : OptiXRapTest.OOtexTest                        ***Exception: SegFault         1.59   
      17 /18  Test #17 : OptiXRapTest.intersect_analytic_test          ***Exception: SegFault         2.22   
      18 /18  Test #18 : OptiXRapTest.Roots3And4Test                   ***Exception: SegFault         1.96   


      13 /18  Test #13 : OptiXRapTest.bufferTest                       Child aborted***Exception:     0.17   
      14 /18  Test #14 : OptiXRapTest.OEventTest                       Child aborted***Exception:     0.46   

      2  /5   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     4.52   
      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     5.47   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     20.82  
    [blyth@localhost opticks]$ 




oxrap tests : Wed
-------------------

::

    [blyth@localhost tests]$ om-test
    === om-test-one : optixrap        /home/blyth/opticks/optixrap                                 /home/blyth/local/opticks/build/optixrap                     
    Wed Apr 10 21:00:59 CST 2019
    Test project /home/blyth/local/opticks/build/optixrap
          Start  1: OptiXRapTest.OContextCreateTest
     1/18 Test  #1: OptiXRapTest.OContextCreateTest ..............   Passed    0.23 sec
          Start  2: OptiXRapTest.OScintillatorLibTest
     2/18 Test  #2: OptiXRapTest.OScintillatorLibTest ............   Passed    0.46 sec
          Start  3: OptiXRapTest.OOTextureTest
     3/18 Test  #3: OptiXRapTest.OOTextureTest ...................   Passed    0.43 sec
          Start  4: OptiXRapTest.OOMinimalTest
     4/18 Test  #4: OptiXRapTest.OOMinimalTest ...................***Exception: SegFault  1.14 sec
          Start  5: OptiXRapTest.OOMinimalRedirectTest
     5/18 Test  #5: OptiXRapTest.OOMinimalRedirectTest ...........***Exception: SegFault  1.21 sec
          Start  6: OptiXRapTest.OOContextTest
     6/18 Test  #6: OptiXRapTest.OOContextTest ...................   Passed    0.39 sec
          Start  7: OptiXRapTest.OOContextUploadDownloadTest
     7/18 Test  #7: OptiXRapTest.OOContextUploadDownloadTest .....   Passed    0.38 sec
          Start  8: OptiXRapTest.LTOOContextUploadDownloadTest
     8/18 Test  #8: OptiXRapTest.LTOOContextUploadDownloadTest ...   Passed    0.38 sec
          Start  9: OptiXRapTest.OOboundaryTest
     9/18 Test  #9: OptiXRapTest.OOboundaryTest ..................   Passed    0.39 sec
          Start 10: OptiXRapTest.OOboundaryLookupTest
    10/18 Test #10: OptiXRapTest.OOboundaryLookupTest ............   Passed    0.44 sec
          Start 11: OptiXRapTest.OOtex0Test
    11/18 Test #11: OptiXRapTest.OOtex0Test ......................***Exception: SegFault  1.16 sec
          Start 12: OptiXRapTest.OOtexTest
    12/18 Test #12: OptiXRapTest.OOtexTest .......................***Exception: SegFault  1.17 sec
          Start 13: OptiXRapTest.bufferTest
    13/18 Test #13: OptiXRapTest.bufferTest ......................Child aborted***Exception:   0.19 sec
          Start 14: OptiXRapTest.OEventTest
    14/18 Test #14: OptiXRapTest.OEventTest ......................Child aborted***Exception:   0.47 sec
          Start 15: OptiXRapTest.OInterpolationTest
    15/18 Test #15: OptiXRapTest.OInterpolationTest ..............   Passed    1.02 sec
          Start 16: OptiXRapTest.ORayleighTest
    16/18 Test #16: OptiXRapTest.ORayleighTest ...................   Passed    1.81 sec
          Start 17: OptiXRapTest.intersect_analytic_test
    17/18 Test #17: OptiXRapTest.intersect_analytic_test .........***Exception: SegFault  1.18 sec
          Start 18: OptiXRapTest.Roots3And4Test
    18/18 Test #18: OptiXRapTest.Roots3And4Test ..................***Exception: SegFault  1.19 sec

    56% tests passed, 8 tests failed out of 18








launch SEGV : OOMinimalTest, OOMinimalRedirectTest, OOtex0Test, OOtexTest, Roots3And4Test
----------------------------------------------------------------------------------------------

::

    2019-04-10 17:25:24.386 INFO  [332047] [OptiXTest::init@39] OptiXTest::init cu minimalTest.cu ptxpath /home/blyth/local/opticks/build/optixrap/OptiXRap_generated_minimalTest.cu.ptx raygen minimal exception exception
    2019-04-10 17:25:24.389 INFO  [332047] [OptiXTest::Summary@72] /home/blyth/local/opticks/lib/OOMinimalTest cu minimalTest.cu ptxpath /home/blyth/local/opticks/build/optixrap/OptiXRap_generated_minimalTest.cu.ptx raygen minimal exception exception
    [New Thread 0x7fff1cff9700 (LWP 332166)]
    [New Thread 0x7ffee9ad5700 (LWP 332179)]

    Program received signal SIGSEGV, Segmentation fault.
    0x00007fffe5b0b387 in ?? () from /lib64/libnvoptix.so.1
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffe5b0b387 in ?? () from /lib64/libnvoptix.so.1
    #1  0x00007fffe5e405d9 in ?? () from /lib64/libnvoptix.so.1
    #2  0x00007fffe5ad9d0e in ?? () from /lib64/libnvoptix.so.1
    #3  0x00007fffe5ada551 in ?? () from /lib64/libnvoptix.so.1
    #4  0x00007fffe5adaffb in ?? () from /lib64/libnvoptix.so.1
    #5  0x00007fffe5ffa094 in ?? () from /lib64/libnvoptix.so.1
    #6  0x00007fffe5f9e996 in ?? () from /lib64/libnvoptix.so.1
    #7  0x0000000000406b13 in optix::ContextObj::launch (this=0x7438b0, entry_point_index=0, image_width=16, image_height=16) at /usr/local/OptiX_600/include/optixu/optixpp_namespace.h:2901
    #8  0x0000000000405969 in main (argc=1, argv=0x7fffffffda48) at /home/blyth/opticks/optixrap/tests/OOMinimalTest.cc:33
    (gdb) exit
    Undefined command: "exit".  Try "help".
    (gdb) quit
    A debugging session is active.




examineBufferFormat assert : OKTest, OKG4Test, OpSeederTest + OEventTest after avoiding version assert
--------------------------------------------------------------------------------------------------------

OKTest and OKG4Test some buffer issue::

    2019-04-10 17:28:01.740 INFO  [336316] [OpticksViz::uploadEvent@357] OpticksViz::uploadEvent (1) DONE 
    2019-04-10 17:28:01.741 INFO  [336316] [OpEngine::uploadEvent@108] .
    OKTest: /home/blyth/opticks/optixrap/OBufBase_.cu:150: void OBufBase::examineBufferFormat(RTformat): Assertion `element_size_bytes == soa*mul' failed.
    Aborted (core dumped)
    [blyth@localhost issues]$ 


OpSeederTest::

    019-04-10 17:29:44.927 ERROR [339099] [OContext::initPrint@131] exit OContext::initPrint with print disabled 
    2019-04-10 17:29:45.102 WARN  [339099] [OGeo::convertMergedMesh@243] OGeo::convertMesh not converting mesh 1 is_null 0 is_skip 0 is_empty 1
    2019-04-10 17:29:46.065 INFO  [339099] [OpticksGen::targetGenstep@303] OpticksGen::targetGenstep setting frame -1 0.0000,0.0000,-0.0000,0.0000 -8914858653937281168777936896.0000,0.0000,-8914858653937281168777936896.0000,0.0000 -0.0000,0.0000,0.0000,0.0000 -0.0000,0.0000,-8956046544105059855626141696.0000,0.0000
    OpSeederTest: /home/blyth/opticks/optixrap/OBufBase_.cu:150: void OBufBase::examineBufferFormat(RTformat): Assertion `element_size_bytes == soa*mul' failed.
    Aborted (core dumped)
    [blyth@localhost issues]$ 

