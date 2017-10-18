Test Fails
=============


Hi Simon,

the installation worked. My final problem concerned xercesx. It was not
installed via the opticks-- . Via xercesx- and and xercesc-make I could install
it but still got the error massage, that
/usr/local/opticks/externals/lib/libxerces-c-3-1.so was not found. The problem
was, that it was not named correctly(libxerces-c-3.1.so). Renaming it solved it
and the installation could be completed.


::

    94% tests passed, 14 tests failed out of 254

    Total Test time (real) = 120.58 sec

    The following tests FAILED:
         30 - BoostRapTest.BOpticksResourceTest (OTHER_FAULT)       # FIXED : handle lack of IDPATH envvar 
        133 - OpticksCoreTest.OpticksCfgTest (SEGFAULT)             # FIXED : uninitialized m_snap_config
        169 - GGeoTest.GPartsTest (OTHER_FAULT)                     # FIXED : GParts requires associated GBndLib to be able to save
        180 - GGeoTest.GMergedMeshTest (OTHER_FAULT)                # FIXED : add protection for zero face mesh (index 1, a skipped mesh?)
        226 - OptiXRapTest.OInterpolationTest (Failed)              # FIXED : with GBndLib::saveAllOverride to save dynamic GBndLib inside TMP, and analysis script path overhaul

        231 - OKOPTest.OpSeederTest (OTHER_FAULT)
        234 - OKOPTest.OpTest (OTHER_FAULT)
        239 - OKTest.VizTest (OTHER_FAULT)
        242 - cfg4Test.CTestDetectorTest (OTHER_FAULT)              # I ALSO SEE THESE FAILS

         85 - NPYTest.NNodeDumpTest (SEGFAULT)
        179 - GGeoTest.GMakerTest (SEGFAULT)
        227 - OptiXRapTest.ORayleighTest (OTHER_FAULT)
        236 - OKTest.OKTest (OTHER_FAULT)
        254 - okg4Test.OKG4Test (OTHER_FAULT)                       # THESE WORK FOR ME

    Errors while running CTest


Darwin::

    97% tests passed, 7 tests failed out of 254

    Total Test time (real) = 130.72 sec

    The following tests FAILED:
        169 - GGeoTest.GPartsTest (OTHER_FAULT)               # now fixed
        180 - GGeoTest.GMergedMeshTest (OTHER_FAULT)          # now fixed
        226 - OptiXRapTest.OInterpolationTest (Failed)        # now fixed

        231 - OKOPTest.OpSeederTest (OTHER_FAULT)             # both buffer control assert 
        239 - OKTest.VizTest (OTHER_FAULT)

        234 - OKOPTest.OpTest (OTHER_FAULT)                   # genstep missing causing OptiX compilation fail
        242 - cfg4Test.CTestDetectorTest (OTHER_FAULT)        

    Errors while running CTest
    opticks-t- : use -V to show output



FIXED : OInterpolationTest : old chestnut, python analysis level missing file GItemList/GBndLib.txt 
-----------------------------------------------------------------------------------------------------

* :doc:`OInterpolationTest_Missing_GBndLib_txt`



OpSeederTest VizTest both assert : OpticksEvent::setBufferControl FATAL: BUFFER LACKS SPEC
-----------------------------------------------------------------------------------------------

::

    2017-10-18 17:53:32.763 INFO  [179839] [SLog::operator@15] OPropagator::OPropagator DONE
    2017-10-18 17:53:32.763 FATAL [179839] [OpticksEvent::setBufferControl@773] OpticksEvent::setBufferControl SKIPPED FOR (null) AS NO spec 
    OpticksEvent::setBufferControl FATAL: BUFFER LACKS SPEC
    Assertion failed: (0), function setBufferControl, file /Users/blyth/opticks/optickscore/OpticksEvent.cc, line 781.
    Abort trap: 6
    simon:opticks blyth$ 


      17          __dd__Geometry__RPC__lvRPCGasgap140xbf98ae0         ce-11611.265 -799018.375 707.900 1434.939 
      18             __dd__Geometry__RPC__lvRPCStrip0xc2213c0         ce-11124.670 -799787.375 707.900 948.345 
      19             __dd__Geometry__RPC__lvRPCStrip0xc2213c0         ce-11263.697 -799567.625 707.900 948.345 
    2017-10-18 17:57:29.903 INFO  [181287] [Composition::setCenterExtent@1000] Composition::setCenterExtent ce -16520.0000,-802110.0000,-7125.0000,7710.5625
    2017-10-18 17:57:29.903 INFO  [181287] [SLog::operator@15] OpticksViz::OpticksViz DONE
    2017-10-18 17:57:29.904 INFO  [181287] [OpticksEvent::addBufferControl@756] OpticksEvent::addBufferControl name seed adding VERBOSE_MODE result: : OPTIX_NON_INTEROP OPTIX_INPUT_ONLY INTEROP_MODE VERBOSE_MODE 
    2017-10-18 17:57:29.904 INFO  [181287] [OpticksEvent::addBufferControl@756] OpticksEvent::addBufferControl name photon adding VERBOSE_MODE result: : OPTIX_OUTPUT_ONLY INTEROP_PTR_FROM_OPENGL INTEROP_MODE VERBOSE_MODE 
    2017-10-18 17:57:29.904 FATAL [181287] [OpticksEvent::setBufferControl@773] OpticksEvent::setBufferControl SKIPPED FOR (null) AS NO spec 
    OpticksEvent::setBufferControl FATAL: BUFFER LACKS SPEC
    Assertion failed: (0), function setBufferControl, file /Users/blyth/opticks/optickscore/OpticksEvent.cc, line 781.
    Abort trap: 6
    simon:opticks blyth$ 


OpTest : failed rtContextCompile from missing genstep_buffer
-----------------------------------------------------------------

::

      18             __dd__Geometry__RPC__lvRPCStrip0xc2213c0         ce-11124.670 -799787.375 707.900 948.345 
      19             __dd__Geometry__RPC__lvRPCStrip0xc2213c0         ce-11263.697 -799567.625 707.900 948.345 
      libc++abi.dylib: terminating with uncaught exception of type optix::Exception: 
      Invalid value (Details: Function "RTresult _rtContextCompile(RTcontext)" caught exception: Initalization of non-primitive type genstep_buffer:  Buffer object, [1769674])
      Abort trap: 6
      simon:opticks blyth$ 


CTestDetectorTest : GSurLib assert 
--------------------------------------


::

    2017-10-18 18:00:08.092 INFO  [182369] [CTraverser::Traverse@128] CTraverser::Traverse DONE
    2017-10-18 18:00:08.092 INFO  [182369] [CTraverser::Summary@104] CDetector::traverse numMaterials 5 numMaterialsWithoutMPT 0
    2017-10-18 18:00:08.092 INFO  [182369] [CDetector::attachSurfaces@240] CDetector::attachSurfaces
    2017-10-18 18:00:08.092 INFO  [182369] [GSurLib::examineSolidBndSurfaces@115] GSurLib::examineSolidBndSurfaces numSolids 7
    2017-10-18 18:00:08.092 FATAL [182369] [GSurLib::examineSolidBndSurfaces@137] GSurLib::examineSolidBndSurfaces i(mm-idx)      6 node(ni.z)      0 node2(id.x)      0 boundary(id.z)      0 parent(ni.w) 4294967295 bname Vacuum///Vacuum lv __dd__Geometry__RPC__lvRPCBarCham140xbf4c6a0
    Assertion failed: (node == i), function examineSolidBndSurfaces, file /Users/blyth/opticks/ggeo/GSurLib.cc, line 147.
    Abort trap: 6
    simon:opticks blyth$ 




