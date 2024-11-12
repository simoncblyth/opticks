FIXED : cxt_min_simtrace_revival
======================================


Issue due to SEvt::gather rejig : had to add it to QSim::simtrace


Issue 1 : After trivial script updates it runs but get no SEvt arrays
----------------------------------------------------------------------

::

    P[blyth@localhost ~]$ ~/o/cxt_min.sh 
                    GEOM : J_2024aug27 
    J_2024aug27_CFBaseFromGEOM : /home/blyth/.opticks/GEOM/J_2024aug27 
                     MOI : uni1:0:0 
                  LOGDIR : /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXTMTest/uni1:0:0 
                    BASE : /data/blyth/opticks/GEOM/J_2024aug27 
            OPTICKS_HASH : 4544b1ae8 
    CUDA_VISIBLE_DEVICES : 1 
                    SDIR : /data/blyth/junotop/opticks/CSGOptiX 
                   SNAME : cxt_min.sh 
                   SSTEM : cxt_min 
                    FOLD : /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXTMTest/uni1:0:0/A000 
                     bin : CSGOptiXTMTest 
                  script :  
                    CEGS : 16:0:9:2000 
              ana_script : /data/blyth/junotop/opticks/CSGOptiX/cxt_min.py 
    /home/blyth/o/cxt_min.sh : run/dbg : delete prior LOGNAME CSGOptiXTMTest.log
    //CSGOptiX7.cu : simtrace idx 0 genstep_id 0 evt->num_simtrace 1254000 
    P[blyth@localhost ~]$ l /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXTMTest/uni1:0:0/A000/
    ls: cannot access /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXTMTest/uni1:0:0/A000/: No such file or directory
    P[blyth@localhost ~]$ l /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXTMTest/
    total 4
    0 drwxrwxr-x.  2 blyth blyth   41 Nov 12 20:16 uni1:0:0
    0 drwxrwxr-x.  3 blyth blyth   22 Nov 12 20:11 .
    4 drwxrwxr-x. 15 blyth blyth 4096 Nov 12 20:11 ..
    P[blyth@localhost ~]$ l /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXTMTest/uni1\:0\:0/
    total 8
    4 -rw-rw-r--. 1 blyth blyth 1363 Nov 12 20:16 run_meta.txt
    4 -rw-rw-r--. 1 blyth blyth  132 Nov 12 20:16 run.npy
    0 drwxrwxr-x. 2 blyth blyth   41 Nov 12 20:16 .
    0 drwxrwxr-x. 3 blyth blyth   22 Nov 12 20:11 ..
    P[blyth@localhost ~]$ 



::

    LOG=1 ~/o/cxt_min.sh    ## looks to be running fine but no SEvt saved 

    P[blyth@localhost CSGOptiX]$ LOG=1 BP=SEvt::SEvt ~/o/cxt_min.sh 

    SLOG::EnvLevel adjusting loglevel by envvar   key CSGOptiX level INFO fallback DEBUG upper_level INFO
    [Detaching after fork from child process 305242]

    Breakpoint 1, 0x00007ffff6edf0f0 in SEvt::SEvt()@plt () from /data/blyth/opticks_Debug/lib/../lib64/libSysRap.so
    (gdb) bt
    #0  0x00007ffff6edf0f0 in SEvt::SEvt()@plt () from /data/blyth/opticks_Debug/lib/../lib64/libSysRap.so
    #1  0x00007ffff6f96b4f in SEvt::Create (ins=0) at /home/blyth/opticks/sysrap/SEvt.cc:1059
    #2  0x00007ffff6f96eb9 in SEvt::CreateOrReuse (idx=0) at /home/blyth/opticks/sysrap/SEvt.cc:1117
    #3  0x00007ffff6f97147 in SEvt::CreateOrReuse () at /home/blyth/opticks/sysrap/SEvt.cc:1161
    #4  0x00007ffff79cb524 in CSGFoundry::AfterLoadOrCreate () at /home/blyth/opticks/CSG/CSGFoundry.cc:3701
    #5  0x00007ffff79c877e in CSGFoundry::Load () at /home/blyth/opticks/CSG/CSGFoundry.cc:3064
    #6  0x00007ffff7bfe362 in CSGOptiX::SimtraceMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:156
    #7  0x0000000000404a75 in main (argc=1, argv=0x7fffffff4448) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXTMTest.cc:13
    (gdb) 



    Thread 1 "CSGOptiXTMTest" hit Breakpoint 1, 0x00007ffff7219e30 in SEvt::beginOfEvent(int)@plt () from /data/blyth/opticks_Debug/lib/../lib64/libQUDARap.so
    (gdb) bt
    #0  0x00007ffff7219e30 in SEvt::beginOfEvent(int)@plt () from /data/blyth/opticks_Debug/lib/../lib64/libQUDARap.so
    #1  0x00007ffff7231702 in QSim::simtrace (this=0x12a26b60, eventID=0) at /home/blyth/opticks/qudarap/QSim.cc:430
    #2  0x00007ffff7c01d77 in CSGOptiX::simtrace (this=0x12a984a0, eventID=0) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:768
    #3  0x00007ffff7bfe387 in CSGOptiX::SimtraceMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:158
    #4  0x0000000000404a75 in main (argc=1, argv=0x7fffffff4448) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXTMTest.cc:13
    (gdb) 

    Thread 1 "CSGOptiXTMTest" hit Breakpoint 1, 0x00007ffff7219790 in SEvt::endOfEvent(int)@plt () from /data/blyth/opticks_Debug/lib/../lib64/libQUDARap.so
    (gdb) bt
    #0  0x00007ffff7219790 in SEvt::endOfEvent(int)@plt () from /data/blyth/opticks_Debug/lib/../lib64/libQUDARap.so
    #1  0x00007ffff723188f in QSim::simtrace (this=0x12a26b60, eventID=0) at /home/blyth/opticks/qudarap/QSim.cc:440
    #2  0x00007ffff7c01d77 in CSGOptiX::simtrace (this=0x12a984a0, eventID=0) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:768
    #3  0x00007ffff7bfe387 in CSGOptiX::SimtraceMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:158
    #4  0x0000000000404a75 in main (argc=1, argv=0x7fffffff4448) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXTMTest.cc:13
    (gdb) 


Save is called but no simtrace::

    2024-11-12 20:39:13.079 INFO  [319155] [CSGOptiX::launch@1091]  (params.width, params.height, params.depth) ( 1920,1080,1) 0.0235
    2024-11-12 20:39:13.079 INFO  [319155] [SEvt::save@3921] SEvt::descComponent
     SEventConfig::GatherCompLabel genstep,simtrace

     SEventConfig::SaveCompLabel genstep,simtrace
                     hit                    - 
                    seed                    - 
                 genstep                    -       SEventConfig::MaxGenstep             3000000
                  photon                    -        SEventConfig::MaxPhoton             3000000
                  record                    -        SEventConfig::MaxRecord                  10
                     aux                    -           SEventConfig::MaxAux                   0
                     sup                    -           SEventConfig::MaxSup                   0
                     rec                    -           SEventConfig::MaxRec                   0
                     seq                    -           SEventConfig::MaxSeq                   1
                  domain                    - 
                simtrace                    - 
                 g4state                    - 
                     pho                    - 
                      gs                    - 



Long ago gather with rejigged, moved upwards::

    3264 /**
    3265 SEvt::gatherComponent
    3266 ------------------------
    3267 
    3268 NB this is for hostside running only, for device-side running 
    3269 the SCompProvider is the QEvent not the SEvt, so this method
    3270 is not called
    3271 
    3272 **/
    3273 
    3274 NP* SEvt::gatherComponent(unsigned cmp) const
    3275 {
    3276     unsigned gather_mask = SEventConfig::GatherComp();
    3277     return gather_mask & cmp ? gatherComponent_(cmp) : nullptr ;
    3278 }



Not called in simtrace running::

    LOG=1 BP=QEvent::gatherComponent ~/o/cxt_min.sh 
    LOG=1 BP=SEvt::gather_components ~/o/cxt_min.sh 
    LOG=1 BP=SEvt::gather ~/o/cxt_min.sh

    3438 /**
    3439 SEvt::gather_components : collects fresh arrays into NPFold from provider
    3440 ---------------------------------------------------------------------------
    3441 
    3442 SEvt::gather_components is invoked by SEvt::gather from SEvt::save::
    3443 
    3444 
    3445      +-------------------+                 +-----------------+
    3446      | QEvent/GPU buf    |                 |  SEvt/NPFold    | 
    3447      |   OR              | === gather ===> |                 |
    3448      | SEvt vecs         |                 |                 |
    3449      +-------------------+                 +-----------------+
    3450 
    3451 
    3452 1. invokes gatherComponent on the SCompProvider instance which is either 
    3453    this SEvt instance for CPU/U4Recorder running OR the QEvent instance
    3454    for GPU/QSim runnning 
    3455 
    3456    * the SCompProvider allocates an NP array and populates it either 
    3457      from vectors for CPU running or by copies from GPU device buffers 
    3458 
    3459 2. the freshly created NP arrays are added to the NPFold, 
    3460    NB pre-existing keys cause NPFold asserts, so it is essential 
    3461    that SEvt::clear is called to clear the fold before gathering 
    3462 
    3463 Note thet QEvent::setGenstep invoked SEvt::clear so the genstep vectors 
    3464 are clear when this gets called. So must rely on the contents of the 
    3465 fold to get the stats. 
    3466 
    3467 **/
    3468 
    3469 void SEvt::gather_components()   // *GATHER*
    3470 {



U4Recorder calls gather from U4Recorder::EndOfEventAction_::

     347 void U4Recorder::EndOfEventAction_(int eventID_)
     348 {
     349     assert( eventID == eventID_ );
     350     LOG_IF(info, SEvt::LIFECYCLE ) << " eventID " << eventID ;
     351 
     352     #if defined(WITH_PMTSIM) && defined(POM_DEBUG)
     353         NP* mtda = PMTSim::ModelTrigger_Debug_Array();
     354         std::string name = mtda->get_meta<std::string>("NAME", "MTDA.npy") ;
     355         sev->add_array( name.c_str(), mtda );
     356     #else
     357         LOG(LEVEL) << "not-(WITH_PMTSIM and POM_DEBUG)"  ;
     358     #endif
     359 
     360     sev->add_array("TRS.npy", U4VolumeMaker::GetTransforms() );
     361     sev->add_array("U4R.npy", MakeMetaArray() );
     362     sev->addEventConfigArray();
     363     sev->gather() ;
     364     sev->endOfEvent(eventID_);  // does save and clear
     365 
     366     const char* savedir = sev->getSaveDir() ;
     367     LOG(LEVEL) << " savedir " << ( savedir ? savedir : "-" );
     368     SaveMeta(savedir);
     369 
     370 }


QSim::

     350 double QSim::simulate(int eventID, bool reset_)
     351 {
     352     SProf::Add("QSim__simulate_HEAD");
     353 
     354     LOG_IF(info, SEvt::LIFECYCLE) << "[ eventID " << eventID ;
     355     if( event == nullptr ) return -1. ;
     356 
     357     sev->beginOfEvent(eventID);  // set SEvt index and tees up frame gensteps for simtrace and input photon simulate running
     358 
     359     int rc = event->setGenstep() ;    // QEvent 
     360     LOG_IF(error, rc != 0) << " QEvent::setGenstep ERROR : have event but no gensteps collected : will skip cx.simulate " ;
     361 
     362 
     363     SProf::Add("QSim__simulate_PREL");
     364 
     365     sev->t_PreLaunch = sstamp::Now() ;
     366     double dt = rc == 0 && cx != nullptr ? cx->simulate_launch() : -1. ;  //SCSGOptiX protocol
     367     sev->t_PostLaunch = sstamp::Now() ;
     368     sev->t_Launch = dt ;
     369 
     370     SProf::Add("QSim__simulate_POST");
     371 
     372     sev->gather();
     373 
     374     SProf::Add("QSim__simulate_DOWN");
     375 
     376     int num_ht = sev->getNumHit() ;   // NB from fold, so requires hits array gathering to be configured to get non-zero 
     377     int num_ph = event->getNumPhoton() ;
     378 
     379     LOG_IF(info, SEvt::MINIMAL)
     380         << " eventID " << eventID
     381         << " dt " << std::setw(11) << std::fixed << std::setprecision(6) << dt
     382         << " ph " << std::setw(10) << num_ph
     383         << " ph/M " << std::setw(10) << num_ph/M
     384         << " ht " << std::setw(10) << num_ht
     385         << " ht/M " << std::setw(10) << num_ht/M
     386         << " reset_ " << ( reset_ ? "YES" : "NO " )
     387         ;
     388 
     389     if(reset_) reset(eventID) ;
     390     SProf::Add("QSim__simulate_TAIL");
     391     return dt ;
     392 }
     393 



::

    P[blyth@localhost ALL0]$ BP=SEvt::gather jok-tds-gdb

    Thread 1 "python" hit Breakpoint 1, 0x00007fffc515e2f0 in SEvt::gather()@plt () from /data/blyth/opticks_Debug/lib64/libQUDARap.so
    (gdb) bt
    #0  0x00007fffc515e2f0 in SEvt::gather()@plt () from /data/blyth/opticks_Debug/lib64/libQUDARap.so
    #1  0x00007fffc517719b in QSim::simulate (this=0x2a20f860, eventID=0, reset_=false) at /home/blyth/opticks/qudarap/QSim.cc:372
    #2  0x00007fffcd2c380a in G4CXOpticks::simulate (this=0xb06f810, eventID=0, reset_=false) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:457
    #3  0x00007fffbddb4e38 in junoSD_PMT_v2_Opticks::EndOfEvent_Simulate (this=0x9ad84c0, eventID=0) at /data/blyth/junotop/junosw/Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2_Opticks.cc:170
    #4  0x00007fffbddb4c64 in junoSD_PMT_v2_Opticks::EndOfEvent (this=0x9ad84c0, eventID=0) at /data/blyth/junotop/junosw/Simulation/DetSimV2/PMTSim/src/junoSD_



Looks like the gather omission in QSim::simtrace, try adding it::

     428 double QSim::simtrace(int eventID)
     429 {
     430     sev->beginOfEvent(eventID);
     431 
     432     int rc = event->setGenstep();
     433     LOG_IF(error, rc != 0) << " QEvent::setGenstep ERROR : no gensteps collected : will skip cx.simtrace " ;
     434 
     435     sev->t_PreLaunch = sstamp::Now() ;
     436     double dt = rc == 0 && cx != nullptr ? cx->simtrace_launch() : -1. ;
     437     sev->t_PostLaunch = sstamp::Now() ;
     438     sev->t_Launch = dt ;
     439 
     440     // see ~/o/notes/issues/cxt_min_simtrace_revival.rst
     441     sev->gather();
     442 
     443     sev->endOfEvent(eventID);
     444 
     445     return dt ;
     446 }


YEP, that succeeds to write some SEvt arrays::

    P[blyth@localhost opticks]$ ./cxt_min.sh 
                    GEOM : J_2024aug27 
    J_2024aug27_CFBaseFromGEOM : /home/blyth/.opticks/GEOM/J_2024aug27 
                     MOI : uni1:0:0 
                     LOG :  
                  LOGDIR : /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXTMTest/uni1:0:0 
                    BASE : /data/blyth/opticks/GEOM/J_2024aug27 
            OPTICKS_HASH : 4544b1ae8 
    CUDA_VISIBLE_DEVICES : 1 
                    SDIR : /data/blyth/junotop/opticks/CSGOptiX 
                   SNAME : cxt_min.sh 
                   SSTEM : cxt_min 
                    FOLD : /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXTMTest/uni1:0:0/A000 
                     bin : CSGOptiXTMTest 
                  script :  
                    CEGS : 16:0:9:2000 
              ana_script : /data/blyth/junotop/opticks/CSGOptiX/cxt_min.py 
    ./cxt_min.sh : run/dbg : delete prior LOGNAME CSGOptiXTMTest.log
    //CSGOptiX7.cu : simtrace idx 0 genstep_id 0 evt->num_simtrace 1254000 
    //CSGOptiX7.cu : simtrace idx 0 pos.xyz 2078.788,675.439,17954.494 mom.xyz   0.151,  0.049,  0.987  
    P[blyth@localhost opticks]$ 
    P[blyth@localhost opticks]$ 
    P[blyth@localhost opticks]$ l /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXTMTest/uni1:0:0/A000/
    total 78452
        0 drwxr-xr-x. 2 blyth blyth      157 Nov 12 21:21 .
        4 -rw-rw-r--. 1 blyth blyth       25 Nov 12 21:21 NPFold_index.txt
        4 -rw-rw-r--. 1 blyth blyth       63 Nov 12 21:21 NPFold_meta.txt
        0 -rw-rw-r--. 1 blyth blyth        0 Nov 12 21:21 NPFold_names.txt
        4 -rw-rw-r--. 1 blyth blyth       88 Nov 12 21:21 sframe_meta.txt
        4 -rw-rw-r--. 1 blyth blyth      384 Nov 12 21:21 sframe.npy
       60 -rw-rw-r--. 1 blyth blyth    60320 Nov 12 21:21 genstep.npy
    78376 -rw-rw-r--. 1 blyth blyth 80256128 Nov 12 21:21 simtrace.npy
        0 drwxrwxr-x. 3 blyth blyth       53 Nov 12 21:21 ..
    P[blyth@localhost opticks]$ 



