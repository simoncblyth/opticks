ggeo-fails-virtualbox-ubuntu_18_04_2
========================================

See vbx-vi for the setup of virtualbox and 
Opticks install inside it.


::

    56% tests passed, 22 tests failed out of 50

    Total Test time (real) =   5.15 sec

    The following tests FAILED:
          4 - GGeoTest.GBufferTest (Child aborted)
          9 - GGeoTest.GItemListTest (Child aborted)
         10 - GGeoTest.GMaterialLibTest (Child aborted)
         13 - GGeoTest.GScintillatorLibTest (Child aborted)
         15 - GGeoTest.GSourceLibTest (Child aborted)
         16 - GGeoTest.GBndLibTest (Child aborted)
         17 - GGeoTest.GBndLibInitTest (Child aborted)
         30 - GGeoTest.GPmtTest (Child aborted)
         31 - GGeoTest.BoundariesNPYTest (Child aborted)
         32 - GGeoTest.GAttrSeqTest (Child aborted)
         33 - GGeoTest.GBBoxMeshTest (Child aborted)
         35 - GGeoTest.GFlagsTest (Child aborted)
         36 - GGeoTest.GGeoLibTest (Child aborted)
         37 - GGeoTest.GGeoTest (Child aborted)
         38 - GGeoTest.GMakerTest (Child aborted)
         39 - GGeoTest.GMergedMeshTest (Child aborted)
         44 - GGeoTest.GPropertyTest (SEGFAULT)
         45 - GGeoTest.GSurfaceLibTest (Child aborted)
         47 - GGeoTest.NLookupTest (Child aborted)
         48 - GGeoTest.RecordsNPYTest (Child aborted)
         49 - GGeoTest.GSceneTest (Child aborted)
         50 - GGeoTest.GMeshLibTest (Child aborted)
    Errors while running CTest
    Thu Apr  4 17:23:20 CST 2019



Looks to all be unsurprising resource fails::


    blyth@blyth-VirtualBox:~/opticks/ggeo$ GBufferTest
    2019-04-04 17:19:55.349 INFO  [6633] [main@89] GBufferTest
    GBufferTest: /home/blyth/opticks/optickscore/OpticksResource.cc:626: void OpticksResource::readEnvironment(): Assertion `daepath' failed.
    Aborted (core dumped)
    blyth@blyth-VirtualBox:~/opticks/ggeo$ GItemListTest
    2019-04-04 17:20:18.182 INFO  [6635] [main@118] GItemListTest
    2019-04-04 17:20:18.184 WARN  [6635] [OpticksResource::readG4Environment@508] OpticksResource::readG4Environment MISSING inipath /usr/local/opticks/externals/config/geant4.ini (create it with bash functions: g4-;g4-export-ini ) 
    2019-04-04 17:20:18.184 WARN  [6635] [OpticksResource::readOpticksEnvironment@532] OpticksResource::readOpticksDataEnvironment MISSING inipath /usr/local/opticks/opticksdata/config/opticksdata.ini (create it with bash functions: opticksdata-;opticksdata-export-ini ) 
    2019-04-04 17:20:18.184 WARN  [6635] [OpticksResource::readEnvironment@607] OpticksResource::readEnvironment NO DAEPATH  geokey OPTICKSDATA_DAEPATH_DYB lastarg NULL daepath NULL
    GItemListTest: /home/blyth/opticks/optickscore/OpticksResource.cc:626: void OpticksResource::readEnvironment(): Assertion `daepath' failed.
    Aborted (core dumped)
    blyth@blyth-VirtualBox:~/opticks/ggeo$ GMaterialLibTest
    2019-04-04 17:20:32.232 WARN  [6637] [OpticksResource::readG4Environment@508] OpticksResource::readG4Environment MISSING inipath /usr/local/opticks/externals/config/geant4.ini (create it with bash functions: g4-;g4-export-ini ) 
    2019-04-04 17:20:32.232 WARN  [6637] [OpticksResource::readOpticksEnvironment@532] OpticksResource::readOpticksDataEnvironment MISSING inipath /usr/local/opticks/opticksdata/config/opticksdata.ini (create it with bash functions: opticksdata-;opticksdata-export-ini ) 
    2019-04-04 17:20:32.232 WARN  [6637] [OpticksResource::readEnvironment@607] OpticksResource::readEnvironment NO DAEPATH  geokey OPTICKSDATA_DAEPATH_DYB lastarg NULL daepath NULL
    GMaterialLibTest: /home/blyth/opticks/optickscore/OpticksResource.cc:626: void OpticksResource::readEnvironment(): Assertion `daepath' failed.
    Aborted (core dumped)
    blyth@blyth-VirtualBox:~/opticks/ggeo$ GScintillatorLibTest
    GScintillatorLibTest: /home/blyth/opticks/optickscore/OpticksResource.cc:626: void OpticksResource::readEnvironment(): Assertion `daepath' failed.
    Aborted (core dumped)
    blyth@blyth-VirtualBox:~/opticks/ggeo$ GSourceLibTest
    GSourceLibTest: /home/blyth/opticks/optickscore/OpticksResource.cc:626: void OpticksResource::readEnvironment(): Assertion `daepath' failed.
    Aborted (core dumped)
    blyth@blyth-VirtualBox:~/opticks/ggeo$ GBndLibTest
    2019-04-04 17:21:03.029 INFO  [6643] [main@52] GBndLibTest
    GBndLibTest: /home/blyth/opticks/optickscore/OpticksResource.cc:626: void OpticksResource::readEnvironment(): Assertion `daepath' failed.
    Aborted (core dumped)
    blyth@blyth-VirtualBox:~/opticks/ggeo$ GBndLibInitTest
    2019-04-04 17:21:13.047 WARN  [6645] [OpticksResource::readG4Environment@508] OpticksResource::readG4Environment MISSING inipath /usr/local/opticks/externals/config/geant4.ini (create it with bash functions: g4-;g4-export-ini ) 
    2019-04-04 17:21:13.048 WARN  [6645] [OpticksResource::readOpticksEnvironment@532] OpticksResource::readOpticksDataEnvironment MISSING inipath /usr/local/opticks/opticksdata/config/opticksdata.ini (create it with bash functions: opticksdata-;opticksdata-export-ini ) 
    2019-04-04 17:21:13.048 WARN  [6645] [OpticksResource::readEnvironment@607] OpticksResource::readEnvironment NO DAEPATH  geokey OPTICKSDATA_DAEPATH_DYB lastarg NULL daepath NULL
    GBndLibInitTest: /home/blyth/opticks/optickscore/OpticksResource.cc:626: void OpticksResource::readEnvironment(): Assertion `daepath' failed.
    Aborted (core dumped)
    blyth@blyth-VirtualBox:~/opticks/ggeo$ GPmtTest
    GPmtTest: /home/blyth/opticks/optickscore/OpticksResource.cc:626: void OpticksResource::readEnvironment(): Assertion `daepath' failed.
    Aborted (core dumped)
    blyth@blyth-VirtualBox:~/opticks/ggeo$ BoundariesNPYTest
    BoundariesNPYTest: /home/blyth/opticks/optickscore/OpticksResource.cc:626: void OpticksResource::readEnvironment(): Assertion `daepath' failed.
    Aborted (core dumped)
    blyth@blyth-VirtualBox:~/opticks/ggeo$ GAttrSeqTest
    2019-04-04 17:21:45.801 WARN  [6651] [OpticksResource::readG4Environment@508] OpticksResource::readG4Environment MISSING inipath /usr/local/opticks/externals/config/geant4.ini (create it with bash functions: g4-;g4-export-ini ) 
    2019-04-04 17:21:45.802 WARN  [6651] [OpticksResource::readOpticksEnvironment@532] OpticksResource::readOpticksDataEnvironment MISSING inipath /usr/local/opticks/opticksdata/config/opticksdata.ini (create it with bash functions: opticksdata-;opticksdata-export-ini ) 
    2019-04-04 17:21:45.802 WARN  [6651] [OpticksResource::readEnvironment@607] OpticksResource::readEnvironment NO DAEPATH  geokey OPTICKSDATA_DAEPATH_DYB lastarg NULL daepath NULL
    GAttrSeqTest: /home/blyth/opticks/optickscore/OpticksResource.cc:626: void OpticksResource::readEnvironment(): Assertion `daepath' failed.
    Aborted (core dumped)
    blyth@blyth-VirtualBox:~/opticks/ggeo$ GBBoxMeshTest
    GBBoxMeshTest: /home/blyth/opticks/optickscore/OpticksResource.cc:626: void OpticksResource::readEnvironment(): Assertion `daepath' failed.
    Aborted (core dumped)
    blyth@blyth-VirtualBox:~/opticks/ggeo$ GFlagsTest
    2019-04-04 17:22:00.585 WARN  [6655] [OpticksResource::readG4Environment@508] OpticksResource::readG4Environment MISSING inipath /usr/local/opticks/externals/config/geant4.ini (create it with bash functions: g4-;g4-export-ini ) 
    2019-04-04 17:22:00.585 WARN  [6655] [OpticksResource::readOpticksEnvironment@532] OpticksResource::readOpticksDataEnvironment MISSING inipath /usr/local/opticks/opticksdata/config/opticksdata.ini (create it with bash functions: opticksdata-;opticksdata-export-ini ) 
    2019-04-04 17:22:00.585 WARN  [6655] [OpticksResource::readEnvironment@607] OpticksResource::readEnvironment NO DAEPATH  geokey OPTICKSDATA_DAEPATH_DYB lastarg NULL daepath NULL
    GFlagsTest: /home/blyth/opticks/optickscore/OpticksResource.cc:626: void OpticksResource::readEnvironment(): Assertion `daepath' failed.
    Aborted (core dumped)
    blyth@blyth-VirtualBox:~/opticks/ggeo$ 



