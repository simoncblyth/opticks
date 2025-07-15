opticks-t-test-fails-with-U4VolumeMaker-PV-GEOM
===================================================


Issue
-------

Geometry created with  g4cx/tests/G4CX_U4TreeCreateCSGFoundryTest.sh lacks
GDML causing two obvious FAILs.  Also QPMTTest fails from lack of PMT info::


    FAILS:  3   / 219   :  Tue Jul 15 13:36:40 2025  :  GEOM BigWaterPool  
      22 /22  Test #22 : QUDARapTest.QPMTTest                                    ***Failed                      0.11   
      9  /30  Test #9  : U4Test.U4GDMLReadTest                                   ***Failed                      0.15   
      20 /30  Test #20 : U4Test.U4TraverseTest                                   ***Failed                      0.15   




::


    22/22 Test #22: QUDARapTest.QPMTTest .....................***Failed    0.11 sec
                    HOME : /home/blyth
                     PWD : /data1/blyth/local/opticks_Debug/build/qudarap/tests
                    GEOM : BigWaterPool
             BASH_SOURCE : /data1/blyth/local/opticks_Debug/bin/QTestRunner.sh
              EXECUTABLE : QPMTTest
                    ARGS : 
    QPMTTest.main.Before:QPMT<float> WITH_CUSTOM4  INSTANCE:NO  
    SPMT::init_total exit as PMTSimParamData is NULL
    SPMT::init_pmtCat expected_type NO  expected_shape NO  pmtCat - total SPMT_Total CD_LPMT:     0 SPMT:     0 WP:     0 ALL:     0
    QPMTTest: /data1/blyth/local/opticks_Debug/include/SysRap/SPMT.h:624: void SPMT::init_pmtCat(): Assertion `expected_type' failed.
    /data1/blyth/local/opticks_Debug/bin/QTestRunner.sh: line 23: 3726963 Aborted                 (core dumped) $EXECUTABLE $@
    /data1/blyth/local/opticks_Debug/bin/QTestRunner.sh : FAIL from QPMTTest




     9/30 Test  #9: U4Test.U4GDMLReadTest .........................***Failed    0.15 sec
                    HOME : /home/blyth
                     PWD : /data1/blyth/local/opticks_Debug/build/u4/tests
                    GEOM : BigWaterPool
             BASH_SOURCE : /data1/blyth/local/opticks_Debug/bin/U4TestRunner.sh
              EXECUTABLE : U4GDMLReadTest
                    ARGS : 
    2025-07-15 13:36:37.708 INFO  [3727071] [main@54]  argv[0] U4GDMLReadTest path /home/blyth/.opticks/GEOM/BigWaterPool/origin.gdml
    2025-07-15 13:36:37.708 FATAL [3727071] [U4GDML::Read@82]  path invalid or does not exist [/home/blyth/.opticks/GEOM/BigWaterPool/origin.gdml]
    /data1/blyth/local/opticks_Debug/bin/U4TestRunner.sh: line 25: 3727071 Segmentation fault      (core dumped) $EXECUTABLE $@
    /data1/blyth/local/opticks_Debug/bin/U4TestRunner.sh : FAIL from U4GDMLReadTest


    20/30 Test #20: U4Test.U4TraverseTest .........................***Failed    0.15 sec
                    HOME : /home/blyth
                     PWD : /data1/blyth/local/opticks_Debug/build/u4/tests
                    GEOM : BigWaterPool
             BASH_SOURCE : /data1/blyth/local/opticks_Debug/bin/U4TestRunner.sh
              EXECUTABLE : U4TraverseTest
                    ARGS : 
    2025-07-15 13:36:38.666 FATAL [3727117] [U4GDML::Read@82]  path invalid or does not exist [/home/blyth/.opticks/GEOM/BigWaterPool/origin.gdml]
    /data1/blyth/local/opticks_Debug/bin/U4TestRunner.sh: line 25: 3727117 Segmentation fault      (core dumped) $EXECUTABLE $@
    /data1/blyth/local/opticks_Debug/bin/U4TestRunner.sh : FAIL from U4TraverseTest






QPMTTest : DONE: find appropriate way to handle lack of PMT info in geom
-------------------------------------------------------------------------

::

    (ok) A[blyth@localhost tests]$ GEOM
    vi /home/blyth/.opticks/GEOM/GEOM.sh
    (ok) A[blyth@localhost tests]$ ./QPMTTest.sh
                               PWD : /home/blyth/opticks/qudarap/tests 
                              FOLD : /data1/blyth/tmp/QPMTTest 
                              GEOM : BigWaterPool 
       BigWaterPool_CFBaseFromGEOM : /home/blyth/.opticks/GEOM/BigWaterPool 
                              name : QPMTTest 
    QPMTTest.main.Before:QPMT<float> WITH_CUSTOM4  INSTANCE:NO  
    SPMT::init_total exit as PMTSimParamData is NULL
    SPMT::init_pmtCat expected_type NO  expected_shape NO  pmtCat - total SPMT_Total CD_LPMT:     0 SPMT:     0 WP:     0 ALL:     0
    QPMTTest: /data1/blyth/local/opticks_Debug/include/SysRap/SPMT.h:624: void SPMT::init_pmtCat(): Assertion `expected_type' failed.
    ./QPMTTest.sh: line 63: 3729302 Aborted                 (core dumped) $name
    ./QPMTTest.sh run error
    (ok) A[blyth@localhost tests]$ 




