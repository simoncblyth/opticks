U4Recorder__EndOfRunAction_Simtrace_FormatIndex_assert
=========================================================

Fixed by allowing prefix  'M' or 'D' in addition to the  normal 'A' and 'B'.
Also revived EndOfRun simtracing.

::

    2025-05-28 11:31:39.279 INFO  [1648989] [U4Recorder::PreUserTrackingAction_Optical@450]  modulo 100000 : ulabel.id 100000
    2025-05-28 11:31:39.785 INFO  [1648989] [U4Recorder::PreUserTrackingAction_Optical@450]  modulo 100000 : ulabel.id 0
    2025-05-28 11:31:53.240 INFO  [1648989] [QSim::simulate@436] sslice {    0,    1,      0,1000000}
    2025-05-28 11:31:53.342 INFO  [1648989] [QEvent::gatherPhoton@613] [ evt.num_photon 1000000 p.sstr (1000000, 4, 4, ) evt.photon 0x7fff9c000000
    [ U4Simtrace::EndOfRunAction
    G4CXTest: /home/blyth/opticks/sysrap/sstr.h:596: static std::string sstr::FormatIndex_(int, char, int, const char*): Assertion `prefix == '\0' || prefix == 'A' || prefix == 'B'' failed.

    Thread 1 "G4CXTest" received signal SIGABRT, Aborted.
    (gdb) bt
    #4  0x00007ffff16373c6 in __assert_fail () from /lib64/libc.so.6
    #5  0x00007ffff2552f79 in sstr::FormatIndex_[abi:cxx11](int, char, int, char const*) (idx=998, prefix=77 'M', wid=3, hdr=0x7ffff267c4bf "SEvt__setIndex_") at /home/blyth/opticks/sysrap/sstr.h:596
    #6  0x00007ffff2543a98 in SEvt::getIndexString_[abi:cxx11](char const*) const (this=0x333f9420, hdr=0x7ffff267c4bf "SEvt__setIndex_") at /home/blyth/opticks/sysrap/SEvt.cc:3882
    #7  0x00007ffff253860a in SEvt::setRunProf_Annotated (this=0x333f9420, hdr=0x7ffff267c4bf "SEvt__setIndex_") at /home/blyth/opticks/sysrap/SEvt.cc:1432
    #8  0x00007ffff253a344 in SEvt::setIndex (this=0x333f9420, index_arg=998) at /home/blyth/opticks/sysrap/SEvt.cc:1866
    #9  0x00007ffff2538b30 in SEvt::beginOfEvent (this=0x333f9420, eventID=998) at /home/blyth/opticks/sysrap/SEvt.cc:1559
    #10 0x00007ffff7e03a96 in U4Simtrace::Scan () at /home/blyth/opticks/u4/U4Simtrace.h:40
    #11 0x00007ffff7e03a3e in U4Simtrace::EndOfRunAction () at /home/blyth/opticks/u4/U4Simtrace.h:27
    #12 0x00007ffff7df8fca in U4Recorder::EndOfRunAction (this=0x6cd2a0) at /home/blyth/opticks/u4/U4Recorder.cc:310
    #13 0x0000000000409f7d in G4CXApp::EndOfRunAction (this=0x6cd200, run=0x42e8b60) at /home/blyth/opticks/g4cx/tests/G4CXApp.h:199
    #14 0x00007ffff77be5ed in G4RunManager::RunTermination() () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Pre-Release/J24.1.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #15 0x00007ffff77bc387 in G4RunManager::BeamOn(int, char const*, int) () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Pre-Release/J24.1.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #16 0x000000000040aa15 in G4CXApp::BeamOn (this=0x6cd200) at /home/blyth/opticks/g4cx/tests/G4CXApp.h:349
    #17 0x000000000040ab21 in G4CXApp::Main () at /home/blyth/opticks/g4cx/tests/G4CXApp.h:356
    #18 0x000000000040acaf in main (argc=1, argv=0x7fffffffb618) at /home/blyth/opticks/g4cx/tests/G4CXTest.cc:13
    (gdb) 

::

    (gdb) f 10
    #10 0x00007ffff7e03a96 in U4Simtrace::Scan () at /home/blyth/opticks/u4/U4Simtrace.h:40
    warning: Source file is more recent than executable.
    40	    evt->beginOfEvent(eventID); 
    (gdb) p eventID
    $1 = 998
    (gdb) f 9
    #9  0x00007ffff2538b30 in SEvt::beginOfEvent (this=0x333f9420, eventID=998) at /home/blyth/opticks/sysrap/SEvt.cc:1559
    1559	    setIndex(eventID);
    (gdb) list
    1554	
    1555	    setStage(SEvt__beginOfEvent);
    1556	    sprof::Stamp(p_SEvt__beginOfEvent_0);
    1557	
    1558	    LOG(LEVEL) << " eventID " << eventID ;   // 0-based
    1559	    setIndex(eventID);
    1560	
    1561	
    1562	    LOG_IF(info, LIFECYCLE) << id() ;
    1563	
    (gdb) f 8
    #8  0x00007ffff253a344 in SEvt::setIndex (this=0x333f9420, index_arg=998) at /home/blyth/opticks/sysrap/SEvt.cc:1866
    1866	    setRunProf_Annotated("SEvt__setIndex_" );
    (gdb) f 7
    #7  0x00007ffff253860a in SEvt::setRunProf_Annotated (this=0x333f9420, hdr=0x7ffff267c4bf "SEvt__setIndex_") at /home/blyth/opticks/sysrap/SEvt.cc:1432
    1432	    std::string eid = getIndexString_(hdr) ;
    (gdb) list
    1427	    SetRunMeta<std::string>( k, sprof::Now() );
    1428	}
    1429	
    1430	void SEvt::setRunProf_Annotated(const char* hdr) const
    1431	{
    1432	    std::string eid = getIndexString_(hdr) ;
    1433	    SetRunMeta<std::string>( eid.c_str(), sprof::Now() );
    1434	}
    1435	
    1436	
    (gdb) f 6
    #6  0x00007ffff2543a98 in SEvt::getIndexString_[abi:cxx11](char const*) const (this=0x333f9420, hdr=0x7ffff267c4bf "SEvt__setIndex_") at /home/blyth/opticks/sysrap/SEvt.cc:3882
    3882	    return sstr::FormatIndex_(index, pfx, wid, hdr );
    (gdb) p index
    $2 = 998
    (gdb) p pfx
    $3 = 77 'M'
    (gdb) p wid
    $4 = 3
    (gdb) p hdr
    $5 = 0x7ffff267c4bf "SEvt__setIndex_"
    (gdb) f 5
    #5  0x00007ffff2552f79 in sstr::FormatIndex_[abi:cxx11](int, char, int, char const*) (idx=998, prefix=77 'M', wid=3, hdr=0x7ffff267c4bf "SEvt__setIndex_") at /home/blyth/opticks/sysrap/sstr.h:596
    596	    assert( prefix == '\0' || prefix == 'A' || prefix == 'B' ); 
    (gdb) 




::

    1860 void SEvt::setIndex(int index_arg)
    1861 {
    1862     assert( index_arg >= 0 );
    1863     index = SEventConfig::EventIndex(index_arg) ;  // may be offset by OPTICKS_START_INDEX
    1864     t_BeginOfEvent = sstamp::Now();                // moved here from the static
    1865 
    1866     setRunProf_Annotated("SEvt__setIndex_" );
    1867 }
    1868 void SEvt::endIndex(int index_arg)
    1869 {
    1870     int index_expected = SEventConfig::EventIndex(index_arg) ;
    1871     bool consistent = index_expected == index ;
    1872     LOG_IF(fatal, !consistent)
    1873          << " index_arg " << index_arg
    1874          << " index_expected " << index_expected
    1875          << " index " << index
    1876          << " consistent " << ( consistent ? "YES" : "NO " )
    1877          ;
    1878     assert( consistent );
    1879     t_EndOfEvent = sstamp::Now();
    1880 
    1881     setRunProf_Annotated("SEvt__endIndex_" );
    1882 }


    1430 void SEvt::setRunProf_Annotated(const char* hdr) const
    1431 {
    1432     std::string eid = getIndexString_(hdr) ;
    1433     SetRunMeta<std::string>( eid.c_str(), sprof::Now() );
    1434 }

    3877 std::string SEvt::getIndexString_(const char* hdr) const
    3878 {
    3879     assert( index >= 0 && index != MISSING_INDEX );
    3880     int wid = 3 ;
    3881     char pfx = getInstancePrefix();
    3882     return sstr::FormatIndex_(index, pfx, wid, hdr );
    3883 }


    0594 inline std::string sstr::FormatIndex_( int idx, char prefix, int wid, const char* hdr )
     595 {
     596     assert( prefix == '\0' || prefix == 'A' || prefix == 'B' );
     597     assert( idx >= 0 );
     598 
     599     std::stringstream ss ;
     600     if(hdr) ss << hdr ;
     601 
     602     //if(prefix) ss << ( idx == 0 ? "z" : ( idx < 0 ? "n" : "p" ) ) ; 
     603     if(prefix != '\0') ss << prefix  ;
     604 
     605     ss << std::setfill('0') << std::setw(wid) << std::abs(idx) ;
     606     std::string str = ss.str();
     607     return str ;
     608 }


Bingo, the simtrace SEvt is not the normal EGPU ECPU currently::

    3864 char SEvt::getInstancePrefix() const
    3865 {
    3866     char pfx = '\0' ;
    3867     switch(instance)
    3868     {
    3869        case EGPU:             pfx = 'A' ; break ;
    3870        case ECPU:             pfx = 'B' ; break ;
    3871        case MISSING_INSTANCE: pfx = 'M' ; break ;
    3872        default:               pfx = 'D' ; break ;
    3873     }
    3874     return pfx ;
    3875 }





BUT simtrace SEvt is written to unexpected dir and missing the simtrace array::

    (ok) A[blyth@localhost tests]$ l /data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/
    total 24
    4 -rw-r--r--. 1 blyth blyth 1873 May 28 13:41 run_meta.txt
    4 -rw-r--r--. 1 blyth blyth  132 May 28 13:41 run.npy
    4 drwxr-xr-x. 5 blyth blyth 4096 May 28 13:31 ..
    4 drwxr-xr-x. 2 blyth blyth 4096 May 28 10:43 A000
    4 drwxr-xr-x. 2 blyth blyth 4096 May 28 10:43 B000

    (ok) A[blyth@localhost tests]$ l /data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/
    total 20
    4 drwxr-xr-x. 3 blyth blyth 4096 May 28 13:31 0
    4 drwxr-xr-x. 4 blyth blyth 4096 May 28 10:43 ALL0_Debug_Philox

    (ok) A[blyth@localhost tests]$ l /data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/0/
    total 20
    4 -rw-r--r--. 1 blyth blyth 1994 May 28 13:41 run_meta.txt
    4 -rw-r--r--. 1 blyth blyth  132 May 28 13:41 run.npy
    4 drwxr-xr-x. 2 blyth blyth 4096 May 28 13:31 M998

    (ok) A[blyth@localhost tests]$ l /data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/0/M998/
    total 20
    0 -rw-r--r--. 1 blyth blyth    0 May 28 13:41 NPFold_index.txt
    4 -rw-r--r--. 1 blyth blyth  557 May 28 13:41 NPFold_meta.txt
    0 -rw-r--r--. 1 blyth blyth    0 May 28 13:41 NPFold_names.txt
    4 -rw-r--r--. 1 blyth blyth  113 May 28 13:41 sframe_meta.txt
    4 -rw-r--r--. 1 blyth blyth  384 May 28 13:41 sframe.npy
    4 drwxr-xr-x. 2 blyth blyth 4096 May 28 13:31 .
    4 drwxr-xr-x. 3 blyth blyth 4096 May 28 13:31 ..


::

     35 inline void U4Simtrace::Scan()
     36 {
     37     int eventID = 998 ;
     38 
     39     SEvt* evt = SEvt::CreateSimtraceEvent();
     40     evt->beginOfEvent(eventID);
     41 
     42     int num_simtrace = int(evt->simtrace.size()) ;
     43 
     44     std::cout
     45         << "U4Simtrace::Scan"
     46         << " num_simtrace " << num_simtrace
     47         << " evt.desc "
     48         << std::endl
     49         << evt->desc()
     50         << std::endl
     51         ;
     52 
     53     bool dump = false ;
     54     for(int i=0 ; i < num_simtrace ; i++)
     55     {
     56         quad4& p = evt->simtrace[i] ;
     57         U4Navigator::Simtrace(p, dump);
     58     }
     59     evt->endOfEvent(eventID);
     60 }




simtrace SEventConfig::EventReldir of "0"  ?
---------------------------------------------

::

    2025-05-28 14:15:09.697 INFO  [1664890] [SEvt::getDir@3858] 
     base_  $TMP/GEOM/$GEOM/$ExecutableName
     SEventConfig::EventReldir   ALL${VERSION:-0}_${OPTICKS_EVENT_NAME:-no_opticks_event_name}
     SEventConfig::_EventReldirDefault ALL${VERSION:-0}_${OPTICKS_EVENT_NAME:-no_opticks_event_name}
     sidx   A000
     path   /data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/A000



    2025-05-28 14:15:20.472 INFO  [1664890] [SEvt::save@3747]  base [$TMP/GEOM/$GEOM/$ExecutableName]
    2025-05-28 14:15:20.472 INFO  [1664890] [SEvt::getDir@3858] 
     base_  $TMP/GEOM/$GEOM/$ExecutableName
     SEventConfig::EventReldir   0
     SEventConfig::_EventReldirDefault ALL${VERSION:-0}_${OPTICKS_EVENT_NAME:-no_opticks_event_name}
     sidx   M998
     path   /data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/0/M998


::

    (ok) A[blyth@localhost sysrap]$ opticks-f SEventConfig::SetEventReldir
    ./sysrap/SEventConfig.cc:void SEventConfig::SetEventReldir(   const char* v){   _EventReldir = v ? strdup(v) : nullptr ; LIMIT_Check() ; }
    ./sysrap/SEvt.cc:        if(frs) SEventConfig::SetEventReldir(frs);
    ./sysrap/SEvt.cc:    if(rel != nullptr) SEventConfig::SetEventReldir(rel);
    ./sysrap/SSimtrace.h:    SEventConfig::SetEventReldir(soname); 
    ./sysrap/tests/SEvtLoadTest.cc:        SEventConfig::SetEventReldir(reldir);  
    ./sysrap/tests/SEvtLoadTest.cc:        SEventConfig::SetEventReldir("SEvtLoadTest"); 
    ./u4/tests/U4AppTest.cc:    SEventConfig::SetEventReldir(desc.c_str() ); 
    (ok) A[blyth@localhost opticks]$ vi 




::

     755 void SEvt::addInputGenstep()
     756 {
     757     LOG_IF(info, LIFECYCLE) << id() ;
     758     LOG(LEVEL);
     759 
     760     if(SEventConfig::IsRGModeSimtrace())
     761     {
     762         const char* frs = frame.get_frs() ; // nullptr when default -1 : meaning all geometry
     763 
     764         LOG_IF(info, SIMTRACE )
     765             << "[" << SEvt__SIMTRACE << "] "
     766             << " frame.get_frs " << ( frs ? frs : "-" ) ;
     767             ;
     768 
     769         //if(frs) SEventConfig::SetEventReldir(frs); // dont do that, default is more standard
     770         // doing this is hangover from separate simtracing of related volumes presumably 
     771 
     772         NP* gs = SFrameGenstep::MakeCenterExtentGenstep_FromFrame(frame);
     773         LOG_IF(info, SIMTRACE) 
     774             << "[" << SEvt__SIMTRACE << "] "
     775             << " simtrace gs " << ( gs ? gs->sstr() : "-" )
     776             ;
     777 
     778         addGenstep(gs);
     779         
     780         if(frame.is_hostside_simtrace()) setFrame_HostsideSimtrace();
     781     }




