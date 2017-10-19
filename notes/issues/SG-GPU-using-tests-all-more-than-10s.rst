SG-GPU-using-tests-all-more-than-10s
=========================================

SG tests that use GPU have a big overhead.

* startup "pedestal" ?

* TODO: see where the times is taken
* try masking out GPUs


::

    203/240 Test #203: CUDARapTest.cudaGetDevicePropertiesTest ......   Passed   10.70 sec
            Start 204: ThrustRapTest.CBufSpecTest
    204/240 Test #204: ThrustRapTest.CBufSpecTest ...................   Passed   10.69 sec
            Start 205: ThrustRapTest.TBufTest
    205/240 Test #205: ThrustRapTest.TBufTest .......................   Passed   10.71 sec
            Start 206: ThrustRapTest.expandTest
    206/240 Test #206: ThrustRapTest.expandTest .....................   Passed   10.70 sec
            Start 207: ThrustRapTest.iexpandTest
    207/240 Test #207: ThrustRapTest.iexpandTest ....................   Passed   10.69 sec
            Start 208: ThrustRapTest.issue628Test
    208/240 Test #208: ThrustRapTest.issue628Test ...................   Passed   10.71 sec
            Start 209: ThrustRapTest.printfTest
    209/240 Test #209: ThrustRapTest.printfTest .....................   Passed   10.70 sec
            Start 210: ThrustRapTest.repeated_rangeTest
    210/240 Test #210: ThrustRapTest.repeated_rangeTest .............   Passed   10.66 sec
            Start 211: ThrustRapTest.strided_rangeTest
    211/240 Test #211: ThrustRapTest.strided_rangeTest ..............   Passed   10.64 sec
            Start 212: ThrustRapTest.strided_repeated_rangeTest
    212/240 Test #212: ThrustRapTest.strided_repeated_rangeTest .....   Passed   10.61 sec
            Start 213: OptiXRapTest.OPropertyLibTest
    213/240 Test #213: OptiXRapTest.OPropertyLibTest ................   Passed   10.39 sec
            Start 214: OptiXRapTest.OScintillatorLibTest
    214/240 Test #214: OptiXRapTest.OScintillatorLibTest ............   Passed   11.75 sec
            Start 215: OptiXRapTest.OOTextureTest
    215/240 Test #215: OptiXRapTest.OOTextureTest ...................   Passed   12.01 sec
            Start 216: OptiXRapTest.OOMinimalTest
    216/240 Test #216: OptiXRapTest.OOMinimalTest ...................   Passed   11.97 sec
            Start 217: OptiXRapTest.OOContextTest
    217/240 Test #217: OptiXRapTest.OOContextTest ...................   Passed   11.83 sec
            Start 218: OptiXRapTest.OOContextUploadDownloadTest
    218/240 Test #218: OptiXRapTest.OOContextUploadDownloadTest .....   Passed   11.91 sec
            Start 219: OptiXRapTest.LTOOContextUploadDownloadTest
    219/240 Test #219: OptiXRapTest.LTOOContextUploadDownloadTest ...   Passed   12.01 sec
            Start 220: OptiXRapTest.OOboundaryTest
    220/240 Test #220: OptiXRapTest.OOboundaryTest ..................   Passed   11.96 sec
            Start 221: OptiXRapTest.OOboundaryLookupTest
    221/240 Test #221: OptiXRapTest.OOboundaryLookupTest ............   Passed   11.97 sec
            Start 222: OptiXRapTest.OOtex0Test
    222/240 Test #222: OptiXRapTest.OOtex0Test ......................   Passed   11.91 sec
            Start 223: OptiXRapTest.OOtexTest
    223/240 Test #223: OptiXRapTest.OOtexTest .......................   Passed   11.89 sec
            Start 224: OptiXRapTest.bufferTest
    224/240 Test #224: OptiXRapTest.bufferTest ......................   Passed   11.94 sec
            Start 225: OptiXRapTest.OEventTest
    225/240 Test #225: OptiXRapTest.OEventTest ......................   Passed   12.23 sec
            Start 226: OptiXRapTest.OInterpolationTest
    226/240 Test #226: OptiXRapTest.OInterpolationTest ..............***Failed   17.39 sec
            Start 227: OptiXRapTest.ORayleighTest
    227/240 Test #227: OptiXRapTest.ORayleighTest ...................***Exception: Other 13.49 sec
            Start 228: OptiXRapTest.intersect_analytic_test
    228/240 Test #228: OptiXRapTest.intersect_analytic_test .........   Passed   12.23 sec
            Start 229: OptiXRapTest.Roots3And4Test
    229/240 Test #229: OptiXRapTest.Roots3And4Test ..................   Passed   12.07 sec
            Start 230: OKOPTest.OpIndexerTest
    230/240 Test #230: OKOPTest.OpIndexerTest .......................   Passed   12.35 sec
            Start 231: OKOPTest.OpSeederTest
    231/240 Test #231: OKOPTest.OpSeederTest ........................   Passed   18.75 sec

    ...

    240/240 Test #240: OKTest.TrivialTest ...........................   Passed    0.02 sec

    99% tests passed, 2 tests failed out of 240

    Total Test time (real) = 482.15 sec

    The following tests FAILED:
        226 - OptiXRapTest.OInterpolationTest (Failed)         ## missing '/tmp/simon/opticks/InterpolationTest/CInterpolationTest_interpol.npy' 
        227 - OptiXRapTest.ORayleighTest (OTHER_FAULT)         ## 
    Errors while running CTest
    opticks-t- : use -V to show output
    [simon@localhost opticks]$ 



OInterpolationTest : missing file 
------------------------------------

* missing '/tmp/simon/opticks/InterpolationTest/CInterpolationTest_interpol.npy' 
* suspect the missing npy comes from cfg4 which is not yet installed on SG


ORayleighTest : rayleigh_buffer dimension ?
-------------------------------------------

* OptiX version issue presumably, SG is with 411 ? D with 380

::

    2017-10-18 20:58:48.396 INFO  [31943] [OContext::close@239] OContext::close setEntryPointCount done.
    2017-10-18 20:58:48.546 INFO  [31943] [OContext::close@245] OContext::close m_cfg->apply() done.
    terminate called after throwing an instance of 'optix::Exception'
      what():  Type mismatch (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" 
      caught exception: Variable "rayleigh_buffer" assigned type Buffer(2d, 16 byte element).  Should be Buffer(1d, 16 byte element).)
    Aborted
    [simon@localhost opticks]$ 





