OptiX_600_CUDA_10.1_test_fails
=================================

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




oxrap tests
--------------

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




unexpected OptiX version : bufferTest, OEventTest
----------------------------------------------------

::

    [blyth@localhost issues]$ bufferTest
    2019-04-10 17:32:51.265 INFO  [343791] [main@106] bufferTest OPTIX_VERSION 60000
    bufferTest: /home/blyth/opticks/optixrap/OConfig.cc:48: static bool OConfig::DefaultWithTop(): Assertion `0 && "unexpected OPTIX_VERSION"' failed.
    Aborted (core dumped)
    [blyth@localhost issues]$ 


::

    2019-04-10 17:31:15.744 ERROR [341425] [OpticksGen::makeLegacyGensteps@194]  code 131072 srctype MACHINERY
    2019-04-10 17:31:15.744 INFO  [341425] [OpticksGen::targetGenstep@303] OpticksGen::targetGenstep setting frame -1 0.0000,0.0000,0.0000,0.0000 -613481534200571583953557782528.0000,-nan,0.0000,0.0000 0.0000,0.0000,0.0000,0.0000 0.0000,0.0000,0.0000,0.0000
    2019-04-10 17:31:15.745 INFO  [341425] [main@41] OEventTest OPTIX_VERSION 60000
    OEventTest: /home/blyth/opticks/optixrap/OConfig.cc:48: static bool OConfig::DefaultWithTop(): Assertion `0 && "unexpected OPTIX_VERSION"' failed.
    Aborted (core dumped)


examineBufferFormat assert : OKTest, OKG4Test, OpSeederTest
--------------------------------------------------------------

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

