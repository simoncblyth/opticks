FIXED SEvt_addGenstep_avoid_nullptr_genstep_warning
======================================================


::

    2338 sgs SEvt::addGenstep(const NP* a)
    2339 {
    2340     sgs s = {} ;
    2341 
    2342     assert(a); // quick way to find whats causing this
    2343     if( a == nullptr )
    2344     {
    2345         LOG(error) << " a null : low level simtrace tests like CSGSimtraceTest.sh can do this  " ;
    2346         return s ;
    2347     }
    2348 


Fix by simple avoidance::

    1062 void SEvt::addInputGenstep()
    1063 {
    1064     LOG_IF(info, LIFECYCLE) << id() ;
    1065     LOG(LEVEL);
    1066 
    1067     NP* igs = createInputGenstep_configured();
    1068     if(igs == nullptr)
    1069     {
    1070         LOG(LEVEL) << "skip addGenstep as igs nullptr" ;
    1071     }
    1072     else
    1073     {
    1074         addGenstep(igs);
    1075     }
    1076 }


So can be more strict in addGenstep from NP array::

    2337 sgs SEvt::addGenstep(const NP* a)
    2338 {
    2339     sgs s = {} ;
    2340 
    2341     LOG_IF(fatal, a == nullptr) << " received nullptr genstep " ;
    2342     NP_FATAL_ASSERT(a);
    2343 
    2344 
    2345     assert( addGenstep_array == 0 );
    2346     addGenstep_array++ ;
    2347 
    2348     int num_gs = a ? a->shape[0] : -1 ;
    2349     assert( num_gs > 0 );
    2350     quad6* qq = (quad6*)a->bytes();
    2351     for(int i=0 ; i < num_gs ; i++) s = addGenstep(qq[i]) ;
    2352 
    2353     if(SEventConfig::IsRGModeSimtrace() && SFrameGenstep::HasConfigEnv()) // CEGS running
    2354     {
    2355         if(fr.is_hostside_simtrace()) setFrame_HostsideSimtrace();
    2356     }
    2357 
    2358     return s ;
    2359 }



Add assert to find cause::


    Begin of Event --> 0
    2026-05-06 16:17:30.524 [junoSD_PMT_v2::EndOfEvent eventID 0 opticksMode 1 hitCollection 0 hitCollectionAlt -1 hcMuon 0 GPU YES
    SProf::Write DISABLED, enable[export SProf__WRITE=1] disable[unset SProf__WRITE]
    python: /home/blyth/opticks/sysrap/SEvt.cc:2342: sgs SEvt::addGenstep(const NP*): Assertion `a' failed.

    Thread 1 "python" received signal SIGABRT, Aborted.
    0x00007ffff748bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff748bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff743eb46 in raise () from /lib64/libc.so.6
    #2  0x00007ffff7428833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff742875b in __assert_fail_base.cold () from /lib64/libc.so.6
    #4  0x00007ffff7437886 in __assert_fail () from /lib64/libc.so.6
    #5  0x00007fffc0720efd in SEvt::addGenstep (this=0xc947aa0, a=0x0) at /home/blyth/opticks/sysrap/SEvt.cc:2342
    #6  0x00007fffc071bee9 in SEvt::addInputGenstep (this=0xc947aa0) at /home/blyth/opticks/sysrap/SEvt.cc:1068
    #7  0x00007fffc071edab in SEvt::beginOfEvent (this=0xc947aa0, eventID=0) at /home/blyth/opticks/sysrap/SEvt.cc:1861
    #8  0x00007fffc0f012b6 in SClientSimulator::simulate (this=0x221750b0, eventID=0, reset=false) at /data1/blyth/local/opticks_Client/include/SysRap/SClientSimulator.h:151
    #9  0x00007fffc0e9d762 in G4CXOpticks::simulate (this=0xc8b9840, eventID=0, reset_=false) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:462
    #10 0x00007fffbf37f122 in junoSD_PMT_v2_Opticks::EndOfEvent_Simulate (this=0xabefbe0, eventID=0) at /home/blyth/junosw/Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2_Opticks.cc:254
    #11 0x00007fffbf37ef3a in junoSD_PMT_v2_Opticks::EndOfEvent (this=0xabefbe0, eventID=0) at /home/blyth/junosw/Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2_Opticks.cc:213
    #12 0x00007fffbf36eae0 in junoSD_PMT_v2::EndOfEvent (this=0x68f07a0, HCE=0xbf590980) at /home/blyth/junosw/Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2.cc:1162
    #13 0x00007fffc4d420fa in G4SDStructure::Terminate(G4HCofThisEvent*) () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.1.1/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4digits_hits.so
    #14 0x00007fffc6c18a80 in G4EventManager::DoProcessing(G4Event*) () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.1.1/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #15 0x00007fffe058468e in G4SvcRunManager::SimulateEvent (this=0x65c1780, i_event=0) at /home/blyth/junosw/Simulation/DetSimV2/G4Svc/src/G4SvcRunManager.cc:29
    #16 0x00007fffce088d2e in DetSimAlg::execute (this=0x66e8890) at /home/blyth/junosw/Simulation/DetSimV2/DetSimAlg/src/DetSimAlg.cc:112
    #17 0x00007fffd24db4b1 in Task::execute() () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.1.1/sniper/InstallArea/lib64/libSniperKernel.so
    #18 0x00007fffd24dfbbd in TaskWatchDog::run() () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.1.1/sniper/InstallArea/lib64/libSniperKernel.so
    #19 0x00007fffd24db054 in Task::run() () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.1.1/sniper/InstallArea/lib64/libSniperKernel.so
    #20 0x00007fffccbd38d3 in boost::python::objects::caller_py_function_impl<boost::python::detail::caller<bool (Task::*)(), boost::python::default_call_policies, boost::mpl::vector2<bool, Task&> > >::operator()(_object*, _object*) () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.1.1/sniper/InstallArea/python/Sniper/libSniperPython.so



Still getting same assert (pilot error - need to build within lo_client)::

    2026-05-06 16:32:24.226 [junoSD_PMT_v2::EndOfEvent eventID 0 opticksMode 1 hitCollection 0 hitCollectionAlt -1 hcMuon 0 GPU YES
    SProf::Write DISABLED, enable[export SProf__WRITE=1] disable[unset SProf__WRITE]
    python: /home/blyth/opticks/sysrap/SEvt.cc:2342: sgs SEvt::addGenstep(const NP*): Assertion `a' failed.

    Thread 1 "python" received signal SIGABRT, Aborted.
    0x00007ffff748bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff748bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff743eb46 in raise () from /lib64/libc.so.6
    #2  0x00007ffff7428833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff742875b in __assert_fail_base.cold () from /lib64/libc.so.6
    #4  0x00007ffff7437886 in __assert_fail () from /lib64/libc.so.6
    #5  0x00007fffc0720efd in SEvt::addGenstep (this=0xc947aa0, a=0x0) at /home/blyth/opticks/sysrap/SEvt.cc:2342
    #6  0x00007fffc071bee9 in SEvt::addInputGenstep (this=0xc947aa0) at /home/blyth/opticks/sysrap/SEvt.cc:1068
    #7  0x00007fffc071edab in SEvt::beginOfEvent (this=0xc947aa0, eventID=0) at /home/blyth/opticks/sysrap/SEvt.cc:1861
    #8  0x00007fffc0f012b6 in SClientSimulator::simulate (this=0x221750b0, eventID=0, reset=false) at /data1/blyth/local/opticks_Client/include/SysRap/SClientSimulator.h:151
    #9  0x00007fffc0e9d762 in G4CXOpticks::simulate (this=0xc8b9840, eventID=0, reset_=false) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:462
    #10 0x00007fffbf37f122 in junoSD_PMT_v2_Opticks::EndOfEvent_Simulate (this=0xabefbe0, eventID=0) at /home/blyth/junosw/Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2_Opticks.cc:254
    #11 0x00007fffbf37ef3a in junoSD_PMT_v2_Opticks::EndOfEvent (this=0xabefbe0, eventID=0) at /home/blyth/junosw/Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2_Opticks.cc:213
    #12 0x00007fffbf36eae0 in junoSD_PMT_v2::EndOfEvent (this=0x68f07a0, HCE=0xbf590980) at /home/blyth/junosw/Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2.cc:1162
    #13 0x00007fffc4d420fa in G4SDStructure::Terminate(G4HCofThisEvent*) () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.1.1/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4digits_hits.so
    #14 0x00007fffc6c18a80 in G4EventManager::DoProcessing(G4Event*) () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.1.1/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #15 0x00007fffe058468e in G4SvcRunManager::SimulateEvent (this=0x65c1780, i_event=0) at /home/blyth/junosw/Simulation/DetSimV2/G4Svc/src/G4SvcRunManager.cc:29
    #16 0x00007fffce088d2e in DetSimAlg::execute (this=0x66e8890) at /home/blyth/junosw/Simulation/DetSimV2/DetSimAlg/src/DetSimAlg.cc:112
    #17 0x00007fffd24db4b1 in Task::execute() () from /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.1.1/sniper/InstallArea/lib64/libSniperKernel.so




