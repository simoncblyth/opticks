FIXED : G4CXTest_raindrop_jpmt_assert
======================================

Fixed by getting the PMT info machinery (SPMT.h,SPMTAccessor.h) 
to work in placeholder fashion when no PMT info paths configured.



::

    (ok) A[blyth@localhost tests]$ ./G4CXTest_raindrop.sh
    ...
    Local_DsG4Scintillation::Local_DsG4Scintillation level 0 verboseLevel 0
    2025-05-28 09:48:35.493 FATAL [1634015] [U4Physics::CreateBoundaryProcess@412]  FAILED TO SPMTAccessor::Load from [CFBaseFromGEOM/CSGFoundry/SSim/extra/jpmt] GEOM RaindropRockAirWater
    G4CXTest: /home/blyth/opticks/u4/U4Physics.cc:417: static G4VProcess* U4Physics::CreateBoundaryProcess(): Assertion `pmt' failed.
    ./G4CXTest_raindrop.sh: line 196: 1634015 Aborted                 (core dumped) $bin
    ./G4CXTest_raindrop.sh : run error

    (ok) A[blyth@localhost tests]$ ./G4CXTest_raindrop.sh dbg
    ...
    Local_DsG4Scintillation::Local_DsG4Scintillation level 0 verboseLevel 0
    2025-05-28 09:51:11.581 FATAL [1634167] [U4Physics::CreateBoundaryProcess@412]  FAILED TO SPMTAccessor::Load from [CFBaseFromGEOM/CSGFoundry/SSim/extra/jpmt] GEOM RaindropRockAirWater
    G4CXTest: /home/blyth/opticks/u4/U4Physics.cc:417: static G4VProcess* U4Physics::CreateBoundaryProcess(): Assertion `pmt' failed.

    (gdb) bt
    ...
    #4  0x00007ffff16373c6 in __assert_fail () from /lib64/libc.so.6
    #5  0x00007ffff7e492eb in U4Physics::CreateBoundaryProcess () at /home/blyth/opticks/u4/U4Physics.cc:417
    #6  0x00007ffff7e48b3e in U4Physics::ConstructOp (this=0x4ea8a0) at /home/blyth/opticks/u4/U4Physics.cc:309
    #7  0x00007ffff7e4759a in U4Physics::ConstructProcess (this=0x4ea8a0) at /home/blyth/opticks/u4/U4Physics.cc:82
    #8  0x00007ffff77ce5fa in G4RunManagerKernel::InitializePhysics() () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Pre-Release/J24.1.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #9  0x00007ffff77bcb43 in G4RunManager::Initialize() () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Pre-Release/J24.1.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #10 0x0000000000409831 in G4CXApp::G4CXApp (this=0x6cd200, runMgr=0x670db0) at /home/blyth/opticks/g4cx/tests/G4CXApp.h:158
    #11 0x000000000040a889 in G4CXApp::Create () at /home/blyth/opticks/g4cx/tests/G4CXApp.h:342
    #12 0x000000000040ab11 in G4CXApp::Main () at /home/blyth/opticks/g4cx/tests/G4CXApp.h:355
    #13 0x000000000040acaf in main (argc=1, argv=0x7fffffffb658) at /home/blyth/opticks/g4cx/tests/G4CXTest.cc:13
    (gdb) 

    (gdb) set listsize 20
    (gdb) list 408
    398	    LOG(LEVEL) << "load path "  << path << " giving PMTSimParamData.data: " << ( data ? "YES" : "NO" ) ; 
    399	    //LOG_IF(LEVEL, data != nullptr ) << *data ; 
    400	
    401	    const PMTAccessor* pmt = PMTAccessor::Create(data) ; 
    402	    const C4IPMTAccessor* ipmt = pmt ;  
    403	    proc = new C4OpBoundaryProcess(ipmt);
    404	
    405	    LOG(LEVEL) << "create C4OpBoundaryProcess :  WITH_CUSTOM4 WITH_PMTSIM " ; 
    406	
    407	#elif defined(WITH_CUSTOM4)
    408	    const char* jpmt = spath::Resolve("$CFBaseFromGEOM/CSGFoundry/SSim/extra/jpmt"); 
    409	
    410	    const SPMTAccessor* pmt = SPMTAccessor::Load(jpmt) ; 
    411	    const char* geom = ssys::getenvvar("GEOM", "no-GEOM") ; 
    412	    LOG_IF(fatal, pmt == nullptr ) 
    413	         << " FAILED TO SPMTAccessor::Load from [" << jpmt << "]" 
    414	         << " GEOM " << ( geom ? geom : "-" )      
    415	         ; 
    416	
    417	    assert(pmt) ;  // trying to get C4 to work without the PMT info, just assert when really need PMT info 
    (gdb) 


Try handling unresolved jpmt, as will happen with raindrop GEOM, with ordinary boundary proc::

    408 #elif defined(WITH_CUSTOM4)
    409     const char* jpmt = spath::Resolve("$CFBaseFromGEOM/CSGFoundry/SSim/extra/jpmt");
    410     bool unresolved = sstr::StartsWith(jpmt,"CFBaseFromGEOM");
    411     if(unresolved)
    412     {
    413         LOG(info) << " WITH_CUSTOM4 BUT jpmt unresolved [" << jpmt << "] so use ordinary G4OpBoundaryProcess" ;
    414         proc = new G4OpBoundaryProcess();
    415     }
    416     else
    417     {


That runs into instrumentation failure, because wrong type of process::

    2025-05-28 10:06:54.353 INFO  [1635156] [U4Recorder::BeginOfEventAction_@333]  eventID 0
    2025-05-28 10:06:54.884 ERROR [1635156] [U4StepPoint::Flag@184]  U4OpBoundaryProcess::GetStatus<T>() : Undefined 
     U4OpBoundaryProcess::Get<T>() NO 
     U4Physics::Switches() 
    U4Physics::Switches
    WITH_CUSTOM4
    NOT:WITH_PMTSIM
    NOT:WITH_CUSTOM4_AND_WITH_PMTSIM
    WITH_CUSTOM4_AND_NOT_WITH_PMTSIM
    NOT:DEBUG_TAG


::

    179     else if( status == fGeomBoundary && proc == U4StepPoint_Transportation )
    180     {
    181         unsigned bstat = U4OpBoundaryProcess::GetStatus<T>();
    182         T* proc = U4OpBoundaryProcess::Get<T>();
    183 
    184         LOG_IF( error, bstat == Undefined )
    185             << " U4OpBoundaryProcess::GetStatus<T>() : Undefined "
    186             << std::endl
    187             << " U4OpBoundaryProcess::Get<T>() " << ( proc ? "YES" : "NO " )
    188             << std::endl
    189             << " U4Physics::Switches() "
    190             << std::endl
    191             << U4Physics::Switches()
    192             ;
    193 
    194         tir = bstat == TotalInternalReflection ;






    2025-05-28 10:06:54.884 ERROR [1635156] [U4StepPoint::Flag@198]  UNEXPECTED BoundaryFlag ZERO  
     flag 0 OpticksPhoton::Flag(flag) .
     bstat 0 U4OpBoundaryProcessStatus::Name(bstat) Undefined
    2025-05-28 10:06:54.884 ERROR [1635156] [U4StepPoint::Flag@184]  U4OpBoundaryProcess::GetStatus<T>() : Undefined 
     U4OpBoundaryProcess::Get<T>() NO 
     U4Physics::Switches() 
    U4Physics::Switches
    WITH_CUSTOM4
    NOT:WITH_PMTSIM
    NOT:WITH_CUSTOM4_AND_WITH_PMTSIM
    WITH_CUSTOM4_AND_NOT_WITH_PMTSIM
    NOT:DEBUG_TAG

    2025-05-28 10:06:54.884 ERROR [1635156] [U4StepPoint::Flag@198]  UNEXPECTED BoundaryFlag ZERO  
     flag 0 OpticksPhoton::Flag(flag) .
     bstat 0 U4OpBoundaryProcessStatus::Name(bstat) Undefined
    2025-05-28 10:06:54.884 ERROR [1635156] [U4Recorder::UserSteppingAction_Optical@1050]  ERR flag zero : post 
    U4StepPoint::DescPositionTime(post)
    U4StepPoint::DescPositionTime (      0.296      0.000     50.000      0.221)
    U4StepPoint::Desc<T>(post)
    U4StepPoint::Desc
     proc 2 procName Transportation procNameRaw Transportation
     status 1 statusName fGeomBoundary
     bstat 0 bstatName Undefined is_tir 0
     flag 0 flagName .
    G4CXTest: /home/blyth/opticks/u4/U4Recorder.cc:1062: void U4Recorder::UserSteppingAction_Optical(const G4Step*) [with T = C4OpBoundaryProcess]: Assertion `flag > 0' failed.
    ./G4CXTest_raindrop.sh: line 196: 1635156 Aborted                 (core dumped) $bin
    ./G4CXTest_raindrop.sh : run error
    (ok) A[blyth@localhost tests]$ 



Need to make C4OpBoundaryProcesss work without the PMT info or with 
placeholder PMT info. 

::

    Thread 1 "G4CXTest" received signal SIGABRT, Aborted.
    #4  0x00007ffff16373c6 in __assert_fail () from /lib64/libc.so.6
    #5  0x00007ffff7e4d02f in SPMT::init_lcqs (this=0x3c7dad0) at /data1/blyth/local/opticks_Debug/include/SysRap/SPMT.h:478
    #6  0x00007ffff7e4c872 in SPMT::init (this=0x3c7dad0) at /data1/blyth/local/opticks_Debug/include/SysRap/SPMT.h:330
    #7  0x00007ffff7e4c7ea in SPMT::SPMT (this=0x3c7dad0, jpmt_=0x0) at /data1/blyth/local/opticks_Debug/include/SysRap/SPMT.h:308
    #8  0x00007ffff7e4c5ab in SPMT::Load (path_=0x3fa01d0 "CFBaseFromGEOM/CSGFoundry/SSim/extra/jpmt") at /data1/blyth/local/opticks_Debug/include/SysRap/SPMT.h:289
    #9  0x00007ffff7e4d7b7 in SPMTAccessor::Load (path=0x3fa01d0 "CFBaseFromGEOM/CSGFoundry/SSim/extra/jpmt") at /data1/blyth/local/opticks_Debug/include/SysRap/SPMTAccessor.h:55
    #10 0x00007ffff7e4916d in U4Physics::CreateBoundaryProcess () at /home/blyth/opticks/u4/U4Physics.cc:410
    #11 0x00007ffff7e48b3e in U4Physics::ConstructOp (this=0x4ea8a0) at /home/blyth/opticks/u4/U4Physics.cc:310
    #12 0x00007ffff7e4759a in U4Physics::ConstructProcess (this=0x4ea8a0) at /home/blyth/opticks/u4/U4Physics.cc:83
    #13 0x00007ffff77ce5fa in G4RunManagerKernel::InitializePhysics() () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Pre-Release/J24.1.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #14 0x00007ffff77bcb43 in G4RunManager::Initialize() () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Pre-Release/J24.1.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #15 0x0000000000409831 in G4CXApp::G4CXApp (this=0x6cd200, runMgr=0x670db0) at /home/blyth/opticks/g4cx/tests/G4CXApp.h:158
    #16 0x000000000040a889 in G4CXApp::Create () at /home/blyth/opticks/g4cx/tests/G4CXApp.h:342
    #17 0x000000000040ab11 in G4CXApp::Main () at /home/blyth/opticks/g4cx/tests/G4CXApp.h:355
    #18 0x000000000040acaf in main (argc=1, argv=0x7fffffffb658) at /home/blyth/opticks/g4cx/tests/G4CXTest.cc:13
    (gdb) 



ana issue, from inconsistent MODE. Fixed by setting MODE in the bash script::

    _poi.shape (100000, 3, 3) 
    pvplt_plotter WSIZE:array([2560, 1440])
    Traceback (most recent call last):
      File "/home/blyth/opticks/g4cx/tests/G4CXTest_raindrop.py", line 147, in <module>
        pl = pvplt_plotter(label)
      File "/home/blyth/opticks/ana/pvplt.py", line 245, in pvplt_plotter
        pl = pv.Plotter(window_size=WSIZE)
             ^^^^^^^^^^
    AttributeError: 'NoneType' object has no attribute 'Plotter'
    ./G4CXTest_raindrop.sh : ana error with script G4CXTest_raindrop.py
    (ok) A[blyth@localhost tests]$ pwd
    /home/blyth/opticks/g4cx/tests



