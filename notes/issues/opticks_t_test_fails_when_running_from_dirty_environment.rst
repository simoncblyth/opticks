opticks_t_test_fails_when_running_from_dirty_environment
==========================================================

Pilot error. Running from a dirty environment, probably from ipc. 
Repeating in fresh tab get no fails.


::


    SLOW: tests taking longer that 15.0 seconds
      27 /43  Test #27 : CSGTest.CSGMakerTest                                    Passed                         31.26  


    FAILS:  3   / 219   :  Thu Jul 10 22:00:40 2025  :  GEOM J25_4_0_opticks_Debug  
      17 /43  Test #17 : CSGTest.CSGFoundry_MakeCenterExtentGensteps_Test        ***Failed                      3.95   
      19 /22  Test #19 : QUDARapTest.QSimWithEventTest                           ***Failed                      3.52   
      20 /22  Test #20 : QUDARapTest.QSimTest                                    ***Failed                      0.57   







::

    9/22 Test #19: QUDARapTest.QSimWithEventTest ............***Failed    3.44 sec
                    HOME : /home/blyth
                     PWD : /data1/blyth/local/opticks_Debug/build/qudarap/tests
                    GEOM : J25_4_0_opticks_Debug
             BASH_SOURCE : /data1/blyth/local/opticks_Debug/bin/QTestRunner.sh
              EXECUTABLE : QSimWithEventTest
                    ARGS : 
    2025-07-10 22:02:16.351 INFO  [3532747] [SEventConfig::SetDevice@1416] SEventConfig::DescDevice
    name                             : NVIDIA RTX 5000 Ada Generation
    totalGlobalMem_bytes             : 33796980736
    totalGlobalMem_GB                : 31
    HeuristicMaxSlot(VRAM)           : 262530128
    HeuristicMaxSlot(VRAM)/M         : 262
    HeuristicMaxSlot_Rounded(VRAM)   : 262000000
    MaxSlot/M                        : 5

    2025-07-10 22:02:16.351 INFO  [3532747] [SEventConfig::SetDevice@1428]  Configured_MaxSlot/M 5 Final_MaxSlot/M 5 HeuristicMaxSlot_Rounded/M 262 changed NO  DeviceName NVIDIA RTX 5000 Ada Generation HasDevice YES
    (export OPTICKS_MAX_SLOT=0 # to use VRAM based HeuristicMaxPhoton) 
    2025-07-10 22:02:16.355 INFO  [3532747] [main@22] [ SSim::Load 
    2025-07-10 22:02:17.055 INFO  [3532747] [main@24] ] SSim::Load : sim 0x141c1e0
    2025-07-10 22:02:17.132 INFO  [3532747] [QRng::initStates@72] initStates<Philox> DO NOTHING : No LoadAndUpload needed  rngmax 1000000000 SEventConfig::MaxCurand 1000000000
    SPMT::init_total SPMT_Total CD_LPMT: 17612 SPMT: 25600 WP:  2400 ALL: 45612
    SPMT::init_pmtCat expected_type YES expected_shape NO  pmtCat (45965, 2, ) total SPMT_Total CD_LPMT: 17612 SPMT: 25600 WP:  2400 ALL: 45612
     main [ SEvt::AddGenstep(gs ) 
    2025-07-10 22:02:17.189 FATAL [3532747] [SEvt::addGenstep@2120] input_photon_with_normal_genstep 1 MIXING input photons with other gensteps is not allowed  for example avoid defining OPTICKS_INPUT_PHOTON when doing simtrace
    QSimWithEventTest: /home/blyth/opticks/sysrap/SEvt.cc:2125: sgs SEvt::addGenstep(const quad6&): Assertion `input_photon_with_normal_genstep == false' failed.
    /data1/blyth/local/opticks_Debug/bin/QTestRunner.sh: line 23: 3532747 Aborted                 (core dumped) $EXECUTABLE $@
    /data1/blyth/local/opticks_Debug/bin/QTestRunner.sh : FAIL from QSimWithEventTest



          Start 20: QUDARapTest.QSimTest
    20/22 Test #20: QUDARapTest.QSimTest .....................***Failed    0.57 sec
                    HOME : /home/blyth
                     PWD : /data1/blyth/local/opticks_Debug/build/qudarap/tests
                    GEOM : J25_4_0_opticks_Debug
             BASH_SOURCE : /data1/blyth/local/opticks_Debug/bin/QTestRunner.sh
              EXECUTABLE : QSimTest
                    ARGS : 
    2025-07-10 22:02:19.798 INFO  [3532820] [main@782] [ TEST WP_PMT_SEMI
    QSimTest: /home/blyth/opticks/qudarap/QSimLaunch.hh:185: static unsigned int QSimLaunch::Type(const char*): Assertion `known' failed.
    /data1/blyth/local/opticks_Debug/bin/QTestRunner.sh: line 23: 3532820 Aborted                 (core dumped) $EXECUTABLE $@
    /data1/blyth/local/opticks_Debug/bin/QTestRunner.sh : FAIL from QSimTest



