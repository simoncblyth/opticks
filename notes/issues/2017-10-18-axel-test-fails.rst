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

        226 - OptiXRapTest.OInterpolationTest (Failed)
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

        226 - OptiXRapTest.OInterpolationTest (Failed)
        231 - OKOPTest.OpSeederTest (OTHER_FAULT)
        234 - OKOPTest.OpTest (OTHER_FAULT)
        239 - OKTest.VizTest (OTHER_FAULT)
        242 - cfg4Test.CTestDetectorTest (OTHER_FAULT)

    Errors while running CTest
    opticks-t- : use -V to show output


GMergedMeshTest
-----------------

::



