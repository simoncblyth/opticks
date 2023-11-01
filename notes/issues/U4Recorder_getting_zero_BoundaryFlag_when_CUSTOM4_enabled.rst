U4Recorder_getting_zero_BoundaryFlag_when_CUSTOM4_enabled
============================================================

Overview
---------

* no such problem without CUSTOM4 compiled in ? 

* probably U4Recorder WITH_CUSTOM4 compiled in is assuming that 
  the WITH_CUSTOM4 physics is in use (specifically C4OpBoundaryProcess) 
  when that is in fact not the case currently in the simple test 



HMM: creating a fully featured C4OpBoundaryProcess needs the 
PMTSimParamSvc/PMTAccessor.h that comes from PMTSim 
as a sneaky way to use junosw functionality inside opticks 

So how can I use the U4Recorder ? Note that the U4Physics 
is only used in testing. 



Issue : U4Recorder getting zero flags at boundaries and tripping assert
--------------------------------------------------------------------------

This issue only happening within junosw+opticks (workstation N) which uses CUSTOM4.
It does not happen within pure opticks (workstation R) with does not use CUSTOM4 currently. 

::

    ~/opticks/g4cx/tests/G4CXTest_raindrop.sh

    ...

    2023-11-01 11:34:37.393 INFO  [87616] [U4Recorder::UserSteppingAction_Optical@852] [  pv drop_pv
    2023-11-01 11:34:37.393 INFO  [87616] [U4Recorder::UserSteppingAction_Optical@875]  first_flag, track 0x3723cf0
    2023-11-01 11:34:37.394 ERROR [87616] [U4StepPoint::Flag@169]  U4OpBoundaryProcess::GetStatus<T>() : Undefined 
     U4OpBoundaryProcess::Get<T>() NO 
     U4Physics::Switches() 
    U4Physics::Switches
    WITH_CUSTOM4
    NOT:WITH_PMTSIM
    NOT:WITH_CUSTOM4_AND_WITH_PMTSIM
    DEBUG_TAG

    2023-11-01 11:34:37.394 ERROR [87616] [U4StepPoint::Flag@181]  UNEXPECTED BoundaryFlag ZERO  
     flag 0 OpticksPhoton::Flag(flag) .
     bstat 0 U4OpBoundaryProcessStatus::Name(bstat) Undefined


    2023-11-01 11:34:37.394 ERROR [87616] [U4StepPoint::Flag@181]  UNEXPECTED BoundaryFlag ZERO  
     flag 0 OpticksPhoton::Flag(flag) .
     bstat 0 U4OpBoundaryProcessStatus::Name(bstat) Undefined
    2023-11-01 11:34:37.394 ERROR [87616] [U4Recorder::UserSteppingAction_Optical@946]  ERR flag zero : post 
    U4StepPoint::DescPositionTime(post)
    U4StepPoint::DescPositionTime (     19.840      0.000    -45.895      0.293)
    U4StepPoint::Desc<T>(post)
    U4StepPoint::Desc
     proc 2 procName Transportation procNameRaw Transportation
     status 1 statusName fGeomBoundary
     bstat 0 bstatName Undefined
     flag 0 flagName .
    G4CXTest: /data/blyth/junotop/opticks/u4/U4Recorder.cc:958: void U4Recorder::UserSteppingAction_Optical(const G4Step*) [with T = C4OpBoundaryProcess]: Assertion `flag > 0' failed.
    ./G4CXTest_raindrop.sh: line 72: 87616 Aborted                 (core dumped) $bin
    ./G4CXTest_raindrop.sh : run error
    N[blyth@localhost tests]$ 




Where the flag comes from
--------------------------

::

    880     unsigned flag = U4StepPoint::Flag<T>(post) ;

::

    163     else if( status == fGeomBoundary && proc == U4StepPoint_Transportation )
    164     {
    165         unsigned bstat = U4OpBoundaryProcess::GetStatus<T>();
    166 
    167         flag = BoundaryFlag(bstat) ;   // BT BR NA SA SD SR DR 
    168 
    169         LOG_IF(error, flag == NAN_ABORT || flag == 0 )
    170             << " UNEXPECTED BoundaryFlag NAN_ABORT/ZERO  "        
    171             << std::endl
    172             << " flag " << flag
    173             << " OpticksPhoton::Flag(flag) " << OpticksPhoton::Flag(flag)
    174             << std::endl
    175             << " bstat " << bstat
    176             << " U4OpBoundaryProcessStatus::Name(bstat) " << U4OpBoundaryProcessStatus::Name(bstat)
    177             ;
    178     }



HMM: creating a fully featured C4OpBoundaryProcess needs the 
PMTSimParamSvc/PMTAccessor.h that comes from PMTSim 
as a sneaky way to use junosw functionality inside opticks 

::

    005 
      6 #if defined(WITH_CUSTOM4) && defined(WITH_PMTSIM)
      7 #include "G4OpBoundaryProcess.hh"
      8 #include "C4OpBoundaryProcess.hh"
      9 #include "PMTSimParamSvc/PMTAccessor.h"
     10 #else
     11 #include "InstrumentedG4OpBoundaryProcess.hh"
     12 #endif
     13 


    264 G4VProcess* U4Physics::CreateBoundaryProcess()  // static 
    265 {
    266     G4VProcess* proc = nullptr ;
    267 
    268 #if defined(WITH_PMTSIM) && defined(WITH_CUSTOM4)
    269     const char* path = "$PMTSimParamData_BASE" ;  // directory with PMTSimParamData subfolder
    270     const PMTSimParamData* data = PMTAccessor::LoadData(path) ;
    271     LOG(LEVEL) << "load path "  << path << " giving PMTSimParamData.data: " << ( data ? "YES" : "NO" ) ;
    272     //LOG_IF(LEVEL, data != nullptr ) << *data ; 
    273 
    274     const PMTAccessor* pmt = PMTAccessor::Create(data) ;
    275     const C4IPMTAccessor* ipmt = pmt ;
    276     proc = new C4OpBoundaryProcess(ipmt);
    277 
    278     LOG(LEVEL) << "create C4OpBoundaryProcess :  WITH_PMTSIM and WITH_CUSTOM4 " ;
    279 #else
    280     proc = new InstrumentedG4OpBoundaryProcess();
    281     LOG(LEVEL) << "create InstrumentedG4OpBoundaryProcess : NOT (WITH_PMTSIM and WITH_CUSTOM4) " ;
    282 #endif
    283     return proc ;
    284 }



