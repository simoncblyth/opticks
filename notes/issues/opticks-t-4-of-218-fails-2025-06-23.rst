opticks-t-4-of-218-fails-2025-06-23
=====================================


Overview
-----------

HMM opticks-t is sensitivite to env of invoking shell, should I move to using an opticks-t.sh 
to avoid the sensitivity ?  NO, opticks-t is inherently tied to build tree and environment.
Instead added opticks-ctest.sh which uses release env. YES: but it is still sensivitive
to invoking env ?


::


    SLOW: tests taking longer that 15.0 seconds
      102/109 Test #102: SysRapTest.SGLFW_SOPTIX_Scene_test                      Passed                         83.58  
      27 /43  Test #27 : CSGTest.CSGMakerTest                                    Passed                         30.29  


    FAILS:  4   / 218   :  Mon Jun 23 14:00:29 2025  :  GEOM J25_4_0_opticks_Debug  
      17 /43  Test #17 : CSGTest.CSGFoundry_MakeCenterExtentGensteps_Test        ***Failed                      3.90   
      29 /43  Test #29 : CSGTest.CSGSimtraceTest                                 ***Failed                      3.63   
      19 /22  Test #19 : QUDARapTest.QSimWithEventTest                           ***Failed                      3.37   
      20 /22  Test #20 : QUDARapTest.QSimTest                                    ***Failed                      0.59   



added opticks-ctest.sh : but same sensitivity
------------------------------------------------

::

    212/215 Test #212: CSGOptiXTest.CSGOptiXRenderTest ..........................   Passed    2.61 sec
            Start 213: CSGOptiXTest.ParamsTest
    213/215 Test #213: CSGOptiXTest.ParamsTest ..................................   Passed    0.48 sec
            Start 214: G4CXTest.G4CXRenderTest
    214/215 Test #214: G4CXTest.G4CXRenderTest ..................................   Passed    2.88 sec
            Start 215: G4CXTest.G4CXOpticks_setGeometry_Test
    215/215 Test #215: G4CXTest.G4CXOpticks_setGeometry_Test ....................   Passed    3.05 sec

    98% tests passed, 4 tests failed out of 215

    Total Test time (real) = 204.87 sec

    The following tests FAILED:
        130 - CSGTest.CSGFoundry_MakeCenterExtentGensteps_Test (Failed)
        142 - CSGTest.CSGSimtraceTest (Failed)
        175 - QUDARapTest.QSimWithEventTest (Failed)
        176 - QUDARapTest.QSimTest (Failed)
    Errors while running CTest
    (ok) A[blyth@localhost tests]$ which opticks-ctest.sh
    /data1/blyth/local/opticks_Debug/bin/opticks-ctest.sh
    (ok) A[blyth@localhost tests]$ 



CSGTest
---------


::

    c
    om-test


    2025-06-23 14:02:57.725 FATAL [3140929] [SEvt::addGenstep@2103] input_photon_with_normal_genstep 1 MIXING input photons with other gensteps is not allowed  for example avoid defining OPTICKS_INPUT_PHOTON when doing simtrace
    CSGFoundry_MakeCenterExtentGensteps_Test: /home/blyth/opticks/sysrap/SEvt.cc:2108: sgs SEvt::addGenstep(const quad6&): Assertion `input_photon_with_normal_genstep == false' failed.
    /data1/blyth/local/opticks_Debug/bin/CSGTestRunner.sh: line 58: 3140929 Aborted                 (core dumped) $EXECUTABLE $@
    /data1/blyth/local/opticks_Debug/bin/CSGTestRunner.sh : FAIL from CSGFoundry_MakeCenterExtentGensteps_Test



    2025-06-23 14:03:45.905 FATAL [3141195] [SEvt::addGenstep@2103] input_photon_with_normal_genstep 1 MIXING input photons with other gensteps is not allowed  for example avoid defining OPTICKS_INPUT_PHOTON when doing simtrace
    CSGSimtraceTest: /home/blyth/opticks/sysrap/SEvt.cc:2108: sgs SEvt::addGenstep(const quad6&): Assertion `input_photon_with_normal_genstep == false' failed.
    /data1/blyth/local/opticks_Debug/bin/CSGTestRunner.sh: line 58: 3141195 Aborted                 (core dumped) $EXECUTABLE $@
    /data1/blyth/local/opticks_Debug/bin/CSGTestRunner.sh : FAIL from CSGSimtraceTest


Cause is running tests in dirty env, dirtied by ipc apparently::

    (ok) A[blyth@localhost CSG]$ env | grep OPTICKS
    OPTICKS_INTEGRATION_MODE=3
    OPTICKS_CUDA_PREFIX=/usr/local/cuda-12.4
    OPTICKS_BUILDTYPE=Debug
    OPTICKS_EVENT_NAME=DebugPhilox_3_PMT_3inch:0:0_RainXZ_Z230_X25_100k_f8
    OPTICKS_HIT_MASK=EC
    OPTICKS_OPTIX_PREFIX=/cvmfs/opticks.ihep.ac.cn/external/OptiX_800
    JUNO_OPTICKS_PREFIX=/data1/blyth/local/opticks_Debug

    OPTICKS_INPUT_PHOTON=RainXZ_Z230_X25_100k_f8.npy
    OPTICKS_INPUT_PHOTON_FRAME=PMT_3inch:0:0

    OPTICKS_MAX_BOUNCE=63
    OPTICKS_HOME=/home/blyth/opticks
    OPTICKS_COMPUTE_ARCHITECTURES=70,89
    OPTICKS_SETUP_VERBOSE=1
    OPTICKS_GEANT4_PREFIX=/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.4.0/ExternalLibs/Geant4/10.04.p02.juno
    OPTICKS_EVENT_MODE=DebugLite
    OPTICKS_IDENTIFIER_NOTE=no uncommited changes and HEAD repo commit matches the last tag hash : so identify with last tag
    OPTICKS_PREFIX=/data1/blyth/local/opticks_Debug
    OPTICKS_MAX_SLOT=M5
    OPTICKS_SCRIPT=InputPhotonsCheck
    OPTICKS_COMPUTE_CAPABILITY=70
    OPTICKS_IDENTIFIER_TIME=20250613165615
    OPTICKS_DOWNLOAD_CACHE=/cvmfs/opticks.ihep.ac.cn/opticks_download_cache
    OPTICKS_STTF_PATH=/data1/blyth/local/opticks_Debug/externals/imgui/imgui/extra_fonts/Cousine-Regular.ttf
    OPTICKS_IDENTIFIER=opticks_v0.4.5
    (ok) A[blyth@localhost CSG]$ 



QUDARapTest
--------------

::

    2025-06-23 14:09:09.198 INFO  [3141719] [QRng::initStates@72] initStates<Philox> DO NOTHING : No LoadAndUpload needed  rngmax 1000000000 SEventConfig::MaxCurand 1000000000
    SPMT::init_total SPMT_Total CD_LPMT: 17612 SPMT: 25600 WP:  2400 ALL: 45612
     main [ SEvt::AddGenstep(gs ) 
    2025-06-23 14:09:09.253 FATAL [3141719] [SEvt::addGenstep@2103] input_photon_with_normal_genstep 1 MIXING input photons with other gensteps is not allowed  for example avoid defining OPTICKS_INPUT_PHOTON when doing simtrace
    QSimWithEventTest: /home/blyth/opticks/sysrap/SEvt.cc:2108: sgs SEvt::addGenstep(const quad6&): Assertion `input_photon_with_normal_genstep == false' failed.
    /data1/blyth/local/opticks_Debug/bin/QTestRunner.sh: line 23: 3141719 Aborted                 (core dumped) $EXECUTABLE $@
    /data1/blyth/local/opticks_Debug/bin/QTestRunner.sh : FAIL from QSimWithEventTest

          Start 20: QUDARapTest.QSimTest
    20/22 Test #20: QUDARapTest.QSimTest .....................***Failed    0.57 sec



