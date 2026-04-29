rejig_effectively_moving_InitEvt_into_SSim
============================================


Rejig
-----

Reposition G4CXOpticks::InitEvt (previously CSGOptiX::InitEvt) functionality:

1. RunMeta metadata collection into SEvt::Init_RUN_META 
2. SEvt hookup with stree into SSim::init



Old position of InitEvt
------------------------

::


    SSimulator* G4CXOpticks::CreateSimulator(CSGFoundry* fd)  // static
    {
        SSimulator* cx = nullptr ;
        //InitEvt(fd->getTree());


    /**
    G4CXOpticks::InitEvt
    ----------------------

    WIP : TRY REPOSITIONING TO LOWER LEVEL

    1. move RunMeta metadata collection into SEvt::Init_RUN_META 
    2. SEvt hookup with stree into SSim::init


    void G4CXOpticks::InitEvt(const stree* tree)  // static
    {
        SEvt* sev = SEvt::CreateOrReuse(SEvt::EGPU) ;
        sev->setTree(tree);

        std::string* rms = SEvt::RunMetaString() ;
        assert(rms);
        bool stamp = false ;
        const char* label = "G4CXOpticks__InitEvt" ;
        smeta::Collect(*rms, label, stamp );
    }
    **/





Causes one  test fail : FIXED BY CHANGING SEventConfig::IntegrationModeDefault to 1 from -1
-----------------------------------------------------------------------------------------------

* maybe when loading from file missed something ?


::

    2026-04-29 15:24:46.141 FATAL [124417] [QEvt::init@130] QEvt instanciated before SEvt instanciated : this is not going to fly 
    G4CXOpticks_setGeometry_Test: /home/blyth/opticks/qudarap/QEvt.cc:132: void QEvt::init(): Assertion `sev' failed.

    Thread 1 "G4CXOpticks_set" received signal SIGABRT, Aborted.
    0x00007ffff128bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff128bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff123eb46 in raise () from /lib64/libc.so.6
    #2  0x00007ffff1228833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff122875b in __assert_fail_base.cold () from /lib64/libc.so.6
    #4  0x00007ffff1237886 in __assert_fail () from /lib64/libc.so.6
    #5  0x00007ffff2abb0ad in QEvt::init (this=0x1824d790) at /home/blyth/opticks/qudarap/QEvt.cc:132
    #6  0x00007ffff2abaf5b in QEvt::QEvt (this=0x1824d790) at /home/blyth/opticks/qudarap/QEvt.cc:114
    #7  0x00007ffff2a7147d in QSim::QSim (this=0x1824d6e0) at /home/blyth/opticks/qudarap/QSim.cc:268
    #8  0x00007ffff2a6f5c5 in QSim::Create () at /home/blyth/opticks/qudarap/QSim.cc:68
    #9  0x00007ffff48a351d in CSGOptiX::InitSim (ssim=0x481ec0) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:273
    #10 0x00007ffff48a3b61 in CSGOptiX::Create (fd=0x14917600) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:350
    #11 0x00007ffff7ed00e4 in G4CXOpticks::CreateSimulator (fd=0x14917600) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:404
    #12 0x00007ffff7ecfbd3 in G4CXOpticks::setGeometry_ (this=0x481e10, fd_=0x14917600) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:361
    #13 0x00007ffff7ecf9e1 in G4CXOpticks::setGeometry (this=0x481e10, fd_=0x14917600) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:330
    #14 0x00007ffff7eceb37 in G4CXOpticks::setGeometry (this=0x481e10) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:228
    #15 0x00007ffff7ecda22 in G4CXOpticks::SetGeometry () at /home/blyth/opticks/g4cx/G4CXOpticks.cc:62
    #16 0x00000000004038c9 in main (argc=1, argv=0x7fffffffba08) at /home/blyth/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.cc:17
    (gdb) 


    (gdb) p sev
    $1 = (SEvt *) 0x0
    (gdb) p SSim::Get()
    $2 = (SSim *) 0x481ec0
    (gdb) p SEvt::Get(0)
    $3 = (SEvt *) 0x0
    (gdb) p SEvt::Get(1)
    $4 = (SEvt *) 0x0


::

     099 QEvt::QEvt()
     100     :
     101     sev(SEvt::Get_EGPU()),
     102     photon_selector(sev ? sev->photon_selector : nullptr),


::

    (gdb) bt
    #0  SSim::SSim (this=0x481e80) at /home/blyth/opticks/sysrap/SSim.cc:171
    #1  0x00007ffff21e6bfc in SSim::Create () at /home/blyth/opticks/sysrap/SSim.cc:97
    #2  0x00007ffff21e63ee in SSim::CreateOrReuse () at /home/blyth/opticks/sysrap/SSim.cc:43
    #3  0x00007ffff7ecdfea in G4CXOpticks::G4CXOpticks (this=0x481e10) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:157
    #4  0x00007ffff7ecda12 in G4CXOpticks::SetGeometry () at /home/blyth/opticks/g4cx/G4CXOpticks.cc:61
    #5  0x00000000004038c9 in main (argc=1, argv=0x7fffffffb9f8) at /home/blyth/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.cc:17
    (gdb) 


    (gdb) bt
    #0  SSim::SSim (this=0x481e80) at /home/blyth/opticks/sysrap/SSim.cc:171
    #1  0x00007ffff21e6bfc in SSim::Create () at /home/blyth/opticks/sysrap/SSim.cc:97
    #2  0x00007ffff21e63ee in SSim::CreateOrReuse () at /home/blyth/opticks/sysrap/SSim.cc:43
    #3  0x00007ffff7ecdfea in G4CXOpticks::G4CXOpticks (this=0x481e10) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:157
    #4  0x00007ffff7ecda12 in G4CXOpticks::SetGeometry () at /home/blyth/opticks/g4cx/G4CXOpticks.cc:61
    #5  0x00000000004038c9 in main (argc=1, argv=0x7fffffffb9f8) at /home/blyth/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.cc:17
    (gdb) c
    Continuing.
     NOT CREATING SEvt : unexpected integrationMode -1

    Breakpoint 1, SSim::SSim (this=0x481ec0) at /home/blyth/opticks/sysrap/SSim.cc:171
    171	    relp(ssys::getenvvar("SSim__RELP", RELP_DEFAULT )),
    (gdb) 







HUH : another FAIL from QSimTest
---------------------------------

Hmm - that test uses SEventConfig statics which means that config
must be done before SEvt instanctiation.


::

      20 /22  Test #20 : QUDARapTest.QSimTest                                    ***Failed                      3.62   


    20/22 Test #20: QUDARapTest.QSimTest .....................***Failed    3.53 sec
                    HOME : /home/blyth
                     PWD : /data1/blyth/local/opticks_Debug/build/qudarap/tests
                    GEOM : J26_1_1_opticks_Debug
             BASH_SOURCE : /data1/blyth/local/opticks_Debug/bin/QTestRunner.sh
              EXECUTABLE : QSimTest
                    ARGS : 
    2026-04-29 15:45:51.423 INFO  [135848] [main@782] [ TEST hemisphere_s_polarized
    2026-04-29 15:45:51.423 INFO  [135848] [main@790] [SSim::Load
    2026-04-29 15:45:51.481 INFO  [135848] [SEventConfig::SetDevice@1893] SEventConfig::DescDevice
    name                             : NVIDIA RTX 5000 Ada Generation
    totalGlobalMem_bytes             : 33770766336
    totalGlobalMem_GB                : 31
    HeuristicMaxSlot(VRAM)           : 257079968
    HeuristicMaxSlot(VRAM)/M         : 257
    HeuristicMaxSlot_Rounded(VRAM)   : 257000000
    MaxSlot/M                        : 0
    ModeLite                         : 0
    ModeMerge                        : 0

    2026-04-29 15:45:51.482 INFO  [135848] [SEventConfig::SetDevice@1908]  Configured_MaxSlot/M 0 Final_MaxSlot/M 257 HeuristicMaxSlot_Rounded/M 257 changed YES DeviceName NVIDIA RTX 5000 Ada Generation HasDevice YES
    (export OPTICKS_MAX_SLOT=0 # to use VRAM based HeuristicMaxPhoton) 
    2026-04-29 15:45:52.593 INFO  [135848] [main@792] ]SSim::Load
    2026-04-29 15:45:52.680 INFO  [135848] [QRng::initStates@72] initStates<Philox> DO NOTHING : No LoadAndUpload needed  rngmax 1000000000000 SEventConfig::MaxCurand 1000000000000
    SPMT::init_pmtNum pmtNum.sstr (6, ) init_pmtNum_FAKE_WHEN_MISSING 1
    [SPMT::init_pmtNum
    [SPMT_Num.desc
     WITH_MPMT : YES
    m_nums_cd_lpmt     :  17612
    m_nums_cd_spmt     :  25600
    m_nums_wp_lpmt     :   2400
    m_nums_wp_atm_lpmt :    348
    m_nums_wp_wal_pmt  :      5
    m_nums_wp_atm_mpmt :    600
    ALL                :  46565
    SUM                :  46565
    SUM==ALL           :    YES
    ]SPMT_Num.desc
    ]SPMT::init_pmtNum
    2026-04-29 15:45:52.753 FATAL [135848] [QSimTest::EventConfig@528] QSimTest::EventConfig must be done prior to instanciating SEvt, eg for fake_propagate bounce consistency 
    QSimTest: /home/blyth/opticks/qudarap/tests/QSimTest.cc:529: static void QSimTest::EventConfig(unsigned int, const SPrd*): Assertion `sev == nullptr' failed.
    /data1/blyth/local/opticks_Debug/bin/QTestRunner.sh: line 23: 135848 Aborted                 (core dumped) $EXECUTABLE $@
    /data1/blyth/local/opticks_Debug/bin/QTestRunner.sh : FAIL from QSimTest


::

    525 void QSimTest::EventConfig(unsigned type, const SPrd* prd )  // static
    526 {
    527     SEvt* sev = SEvt::Get_EGPU();
    528     LOG_IF(fatal, sev != nullptr ) << "QSimTest::EventConfig must be done prior to instanciating SEvt, eg for fake_propagate bounce consistency " ;
    529     assert(sev == nullptr);
    530 
    531     LOG(LEVEL) << "[ " <<  QSimLaunch::Name(type) ;
    532     if( type == FAKE_PROPAGATE )
    533     {
    534         LOG(LEVEL) << prd->desc() ;
    535         int maxbounce = prd->getNumBounce();
    536 
    537         SEventConfig::SetMaxBounce(maxbounce);
    538         SEventConfig::SetEventMode("DebugLite");
    539         SEventConfig::Initialize();
    540 
    541         SEventConfig::SetMaxGenstep(1);    // FAKE_PROPAGATE starts from input photons but uses a single placeholder genstep
    542 
    543         unsigned mx = 1000000 ;
    544         SEventConfig::SetMaxPhoton(mx);   // used for QEvt buffer sizing
    545         SEventConfig::SetMaxSlot(mx);
    546         // greatly reduced MaxSlot as debug arrays in use
    547 
    548         LOG(LEVEL) << " SEventConfig::Desc " << SEventConfig::Desc() ;
    549     }
    550     LOG(LEVEL) << "] " <<  QSimLaunch::Name(type) ;
    551 }



Chicken-and-egg problem with QSimTest::

    777 int main(int argc, char** argv)
    778 {
    779     OPTICKS_LOG(argc, argv);
    780 
    781     const char* TEST = ssys::getenvvar("TEST", "hemisphere_s_polarized");
    782     LOG(info) << "[ TEST " << TEST ;
    783 
    784 
    785     int type = QSimLaunch::Type(TEST);
    786     if(type == UNKNOWN) return 0 ;
    787 
    788     unsigned num = QSimTest::Num(argc, argv);
    789 
    790     LOG(info) << "[SSim::Load" ;
    791     SSim* sim = SSim::Load();
    792     LOG(info) << "]SSim::Load" ;
    793     assert(sim);
    794 
    795 
    796     QSim::UploadComponents(sim);   // instanciates things like QBnd : NORMALLY FIRST GPU ACCESS
    797     const SPrd* prd = sim->get_sprd() ;
    798 
    799     LOG_IF(error, prd->rc != 0 )
    800         << " SPrd::rc NON-ZERO " << prd->rc
    801         << " NOT ALL CONFIGURED BOUNDARIES ARE IN THE GEOMETRY "
    802         << "\nprd.desc\n"
    803         << prd->desc()
    804         << "\nsim.desc\n"
    805         << sim->desc()
    806         ;
    807     if(prd->rc != 0 ) return 0 ; // avoid test fail when using geometry without expected boundaries
    808 
    809 
    810     QSimTest::EventConfig(type, prd );  // must be after QBnd instanciation and before SEvt instanciation
    811 
    812     [[maybe_unused]] SEvt* ev = SEvt::Create_EGPU() ;
    813     assert(ev);
    814 
    815 
    816     QSimTest qst(type, num, prd)  ;
    817     qst.main();
    818 
    819     cudaDeviceSynchronize();
    820 
    821     LOG(info) << "] TEST " << TEST << " qst.rc " << qst.rc ;
    822     return qst.rc  ;
    823 }


Break the lock by adding SPrd::Load which can be used before SSim::Load.

