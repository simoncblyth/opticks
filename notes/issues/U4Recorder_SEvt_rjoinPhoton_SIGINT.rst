U4Recorder_SEvt_rjoinPhoton_SIGINT
===================================

::

    2023-12-09 15:53:57.179 INFO  [260656] [G4CXApp::BeamOn@342] [ OPTICKS_NUM_EVENT=1
    2023-12-09 15:55:27.034 INFO  [260656] [G4CXApp::GeneratePrimaries@223] [ SEventConfig::RunningModeLabel SRM_TORCH eventID 0
    U4VPrimaryGenerator::GeneratePrimaries_From_Photons ph (1000000, 4, 4, )
    2023-12-09 15:55:27.748 INFO  [260656] [G4CXApp::GeneratePrimaries@253] ]  eventID 0
    2023-12-09 15:55:27.748 INFO  [260656] [U4Recorder::BeginOfEventAction_@292]  eventID 0
    2023-12-09 15:55:28.176 INFO  [260656] [SEvt::hostside_running_resize_@2235]  photon.size 0 photon.size/M 0 =>  evt.num_photon 1000000 evt.num_photon/M 1

    Thread 1 "G4CXTest" received signal SIGINT, Interrupt.
    0x00007ffff366b4fb in raise () from /lib64/libpthread.so.0
    (gdb) bt
    #0  0x00007ffff366b4fb in raise () from /lib64/libpthread.so.0
    #1  0x00007ffff3e0fab4 in SEvt::rjoinPhoton (this=0x6c5e90, label=...) at /home/blyth/junotop/opticks/sysrap/SEvt.cc:2552
    #2  0x00007ffff7cf9d33 in U4Recorder::PreUserTrackingAction_Optical (this=0x6c5c90, track=0x4f760cf0) at /home/blyth/junotop/opticks/u4/U4Recorder.cc:424
    #3  0x00007ffff7cf9526 in U4Recorder::PreUserTrackingAction (this=0x6c5c90, track=0x4f760cf0) at /home/blyth/junotop/opticks/u4/U4Recorder.cc:332
    #4  0x000000000040a3ef in G4CXApp::PreUserTrackingAction (this=0x6c5bf0, trk=0x4f760cf0) at /home/blyth/junotop/opticks/g4cx/tests/G4CXApp.h:306
    #5  0x00007ffff7036c20 in G4TrackingManager::ProcessOneTrack(G4Track*) ()
       from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #6  0x00007ffff7071d0d in G4EventManager::DoProcessing(G4Event*) ()
       from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #7  0x00007ffff7111a9f in G4RunManager::DoEventLoop(int, char const*, int) ()
       from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #8  0x00007ffff710f4de in G4RunManager::BeamOn(int, char const*, int) ()
       from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #9  0x000000000040a8cd in G4CXApp::BeamOn (this=0x6c5bf0) at /home/blyth/junotop/opticks/g4cx/tests/G4CXApp.h:343
    #10 0x000000000040a9d9 in G4CXApp::Main () at /home/blyth/junotop/opticks/g4cx/tests/G4CXApp.h:350
    #11 0x000000000040ab63 in main (argc=1, argv=0x7fffffff2298) at /home/blyth/junotop/opticks/g4cx/tests/G4CXTest.cc:16
    (gdb) 

::

    (gdb) f 3
    #3  0x00007ffff7cf9526 in U4Recorder::PreUserTrackingAction (this=0x6c5c90, track=0x4f760cf0) at /home/blyth/junotop/opticks/u4/U4Recorder.cc:332
    332	void U4Recorder::PreUserTrackingAction(const G4Track* track){  LOG(LEVEL) ; if(U4Track::IsOptical(track)) PreUserTrackingAction_Optical(track); }
    (gdb) f 2
    #2  0x00007ffff7cf9d33 in U4Recorder::PreUserTrackingAction_Optical (this=0x6c5c90, track=0x4f760cf0) at /home/blyth/junotop/opticks/u4/U4Recorder.cc:424
    424	            sev->rjoinPhoton(ulabel); 
    (gdb) f 1
    #1  0x00007ffff3e0fab4 in SEvt::rjoinPhoton (this=0x6c5e90, label=...) at /home/blyth/junotop/opticks/sysrap/SEvt.cc:2552
    2552	    if(parent_idx_expect) std::raise(SIGINT); 
    (gdb) 




