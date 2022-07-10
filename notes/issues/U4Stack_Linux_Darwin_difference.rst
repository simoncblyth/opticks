U4Stack_Linux_Darwin_difference
==================================

Linux SBacktrace::Summary missing crucial line with "DsG4Scintillation::ResetNumberOfInteractionLengthLeft"
But scintillation is the only G4VRestDiscreteProcess in play so do not need to Shim.

But same problem with other processes. 

The deficient SBacktrace::Summary might arise from the inline imps.



::

    class DsG4Scintillation   : public G4VRestDiscreteProcess, public G4UImessenger

    class G4VRestDiscreteProcess : public G4VProcess 

    class G4VDiscreteProcess     : public G4VProcess


    class G4OpAbsorption      : public G4VDiscreteProcess

    class G4OpRayleigh        : public G4VDiscreteProcess

    class G4OpBoundaryProcess : public G4VDiscreteProcess






::

    2022-07-10 22:05:00.704 INFO  [454628] [U4RecorderTest::GeneratePrimaries@134] ]
    2022-07-10 22:05:00.704 INFO  [454628] [U4Recorder::BeginOfEventAction@77] 
    2022-07-10 22:05:00.718 ERROR [454628] [U4Random::flat@431] 
    SBacktrace::Summary
    U4Random::flat
    G4VRestDiscreteProcess::PostStepGetPhysicalInteractionLength
    G4SteppingManager::DefinePhysicalStepLength
    G4SteppingManager::Stepping
    G4TrackingManager::ProcessOneTrack
    G4EventManager::DoProcessing
    G4RunManager::DoEventLoop
    G4RunManager::BeamOn

    2022-07-10 22:05:00.718 INFO  [454628] [U4Random::flat@438] U4Random_select - m_select->size 0

    Program received signal SIGINT, Interrupt.
    0x00007ffff09484fb in raise () from /lib64/libpthread.so.0
    Missing separate debuginfos, use: debuginfo-install bzip2-libs-1.0.6-13.el7.x86_64 cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 freetype-2.8-12.el7_6.1.x86_64 glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libICE-1.0.9-9.el7.x86_64 libSM-1.2.2-2.el7.x86_64 libX11-1.6.7-4.el7_9.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-59.el7_9.1.x86_64 libicu-50.2-4.el7_7.x86_64 libidn-1.28-4.el7.x86_64 libpng-1.5.13-7.el7_2.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libuuid-2.23.2-59.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-25.el7_9.x86_64 openssl-libs-1.0.2k-24.el7_9.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007ffff09484fb in raise () from /lib64/libpthread.so.0
    #1  0x00007ffff7ba9090 in U4Random::flat (this=0x7fffffff5900) at /data/blyth/junotop/opticks/u4/U4Random.cc:441
    #2  0x00000000004267b2 in DsG4Scintillation::ResetNumberOfInteractionLengthLeft (this=0x1c80150) at /data/blyth/junotop/opticks/u4/tests/DsG4Scintillation.cc:114
    #3  0x00007ffff37e8554 in G4VRestDiscreteProcess::PostStepGetPhysicalInteractionLength(G4Track const&, double, G4ForceCondition*) ()
       from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4processes.so
    #4  0x00007ffff44ae599 in G4SteppingManager::DefinePhysicalStepLength() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #5  0x00007ffff44acb48 in G4SteppingManager::Stepping() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #6  0x00007ffff44b8472 in G4TrackingManager::ProcessOneTrack(G4Track*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #7  0x00007ffff46ef389 in G4EventManager::DoProcessing(G4Event*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #8  0x00007ffff498aa6f in G4RunManager::DoEventLoop(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #9  0x00007ffff498853e in G4RunManager::BeamOn(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #10 0x0000000000413a97 in main (argc=1, argv=0x7fffffff6308) at /data/blyth/junotop/opticks/u4/tests/U4RecorderTest.cc:196
    (gdb) 



::

    (gdb) c
    Continuing.
    2022-07-10 22:21:45.729 ERROR [454628] [U4Random::flat@431] 
    SBacktrace::Summary
    U4Random::flat
    G4SteppingManager::DefinePhysicalStepLength
    G4SteppingManager::Stepping
    G4TrackingManager::ProcessOneTrack
    G4EventManager::DoProcessing
    G4RunManager::DoEventLoop
    G4RunManager::BeamOn

    2022-07-10 22:21:45.729 INFO  [454628] [U4Random::flat@438] U4Random_select - m_select->size 0

    Program received signal SIGINT, Interrupt.
    0x00007ffff09484fb in raise () from /lib64/libpthread.so.0
    (gdb) bt
    #0  0x00007ffff09484fb in raise () from /lib64/libpthread.so.0
    #1  0x00007ffff7ba9090 in U4Random::flat (this=0x7fffffff5900) at /data/blyth/junotop/opticks/u4/U4Random.cc:441
    #2  0x000000000041c536 in ShimG4OpRayleigh::ResetNumberOfInteractionLengthLeft (this=0x1cdbd00) at /data/blyth/junotop/opticks/u4/ShimG4OpRayleigh.h:48
    #3  0x000000000041c656 in ShimG4OpRayleigh::PostStepGetPhysicalInteractionLength (this=0x1cdbd00, track=..., previousStepSize=0, condition=0xa4fef8)
        at /data/blyth/junotop/opticks/u4/ShimG4OpRayleigh.h:73
    #4  0x00007ffff44ae599 in G4SteppingManager::DefinePhysicalStepLength() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #5  0x00007ffff44acb48 in G4SteppingManager::Stepping() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #6  0x00007ffff44b8472 in G4TrackingManager::ProcessOneTrack(G4Track*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #7  0x00007ffff46ef389 in G4EventManager::DoProcessing(G4Event*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #8  0x00007ffff498aa6f in G4RunManager::DoEventLoop(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #9  0x00007ffff498853e in G4RunManager::BeamOn(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #10 0x0000000000413a97 in main (argc=1, argv=0x7fffffff6308) at /data/blyth/junotop/opticks/u4/tests/U4RecorderTest.cc:196
    (gdb) 








::

    022-07-10 23:23:15.135 INFO  [456353] [U4Recorder::BeginOfEventAction@77] 
    2022-07-10 23:23:15.149 ERROR [456353] [U4Random::flat@431] 
    SBacktrace::Summary
    U4Random::flat
    G4VRestDiscreteProcess::PostStepGetPhysicalInteractionLength
    G4SteppingManager::DefinePhysicalStepLength
    G4SteppingManager::Stepping
    G4TrackingManager::ProcessOneTrack
    G4EventManager::DoProcessing
    G4RunManager::DoEventLoop
    G4RunManager::BeamOn

    2022-07-10 23:23:15.150 INFO  [456353] [U4Random::flat@438] U4Random_select - m_select->size 0

    Program received signal SIGINT, Interrupt.
    0x00007ffff09484fb in raise () from /lib64/libpthread.so.0
    Missing separate debuginfos, use: debuginfo-install bzip2-libs-1.0.6-13.el7.x86_64 cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 freetype-2.8-12.el7_6.1.x86_64 glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libICE-1.0.9-9.el7.x86_64 libSM-1.2.2-2.el7.x86_64 libX11-1.6.7-4.el7_9.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-59.el7_9.1.x86_64 libicu-50.2-4.el7_7.x86_64 libidn-1.28-4.el7.x86_64 libpng-1.5.13-7.el7_2.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libuuid-2.23.2-59.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-25.el7_9.x86_64 openssl-libs-1.0.2k-24.el7_9.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007ffff09484fb in raise () from /lib64/libpthread.so.0
    #1  0x00007ffff7b9fa0c in U4Random::flat (this=0x7fffffff57c0) at /data/blyth/junotop/opticks/u4/U4Random.cc:441
    #2  0x0000000000423a36 in DsG4Scintillation::ResetNumberOfInteractionLengthLeft (this=0x1c7ec00) at /data/blyth/junotop/opticks/u4/tests/DsG4Scintillation.cc:114
    #3  0x00007ffff37e8554 in G4VRestDiscreteProcess::PostStepGetPhysicalInteractionLength(G4Track const&, double, G4ForceCondition*) ()
       from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4processes.so
    #4  0x00007ffff44ae599 in G4SteppingManager::DefinePhysicalStepLength() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #5  0x00007ffff44acb48 in G4SteppingManager::Stepping() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #6  0x00007ffff44b8472 in G4TrackingManager::ProcessOneTrack(G4Track*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #7  0x00007ffff46ef389 in G4EventManager::DoProcessing(G4Event*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #8  0x00007ffff498aa6f in G4RunManager::DoEventLoop(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #9  0x00007ffff498853e in G4RunManager::BeamOn(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #10 0x0000000000413951 in main (argc=1, argv=0x7fffffff61c8) at /data/blyth/junotop/opticks/u4/tests/U4RecorderTest.cc:200
    (gdb) 




Switch to manual tagging : occasionally missing a consumption
----------------------------------------------------------------------

::

    pre  U4StepPoint::DescPositionTime (     12.745     -7.456   -990.000      0.000)
     post U4StepPoint::DescPositionTime (     12.745     -7.456   1629.730     13.437)
    2022-07-11 01:16:39.641 ERROR [12213] [U4Random::check_cursor_vs_tagslot@487]  m_seq_index 9956 cursor 53 slot 52 cursor_slot_match 0
     PROBABLY SOME RANDOM CONSUMPTION LACKS SEvt::AddTag CALLS 
    U4Step::MockOpticksBoundaryIdentity problem step  idx 9955 type 5 U4Step::Name UNEXPECTED cosThetaSign 0 spec LS///LS boundary 4294967295 kludge_prim_idx 0
     pre  U4StepPoint::DescPositionTime (    -10.187    -10.697   -990.000      0.000)
     post U4StepPoint::DescPositionTime (    -10.187    -10.697  17824.000     96.501)
    U4Step::MockOpticksBoundaryIdentity problem step  idx 9955 type 1 U4Step::Name NOT_AT_BOUNDARY cosThetaSign 0 spec  boundary 0 kludge_prim_idx 0
     pre  U4StepPoint::DescPositionTime (    -10.187    -10.697  17824.00


    post U4StepPoint::DescPositionTime (      0.700    -38.031  48749.974    209.862)
    U4Step::MockOpticksBoundaryIdentity problem step  idx 9920 type 1 U4Step::Name NOT_AT_BOUNDARY cosThetaSign 0 spec  boundary 0 kludge_prim_idx 0
     pre  U4StepPoint::DescPositionTime (      8.095     -1.973   -990.000      0.000)
     post U4StepPoint::DescPositionTime (      8.095     -1.973   2913.424     20.021)
    2022-07-11 01:16:39.699 ERROR [12213] [U4Random::check_cursor_vs_tagslot@487]  m_seq_index 9920 cursor 53 slot 52 cursor_slot_match 0
     PROBABLY SOME RANDOM CONSUMPTION LACKS SEvt::AddTag CALLS 
    U4Step::MockOpticksBoundaryIdentity problem step  idx 9919 type 1 U4Step::Name NOT_AT_BOUNDARY cosThetaSign 0 spec  boundary 0 kludge_prim_idx 0
     pre  U4StepPoint::DescPositionTime (      1.424    -25.395   -990.000      0.000)
     post U4StepPoint::DescPositionTime (      1.424    -25.395  14759.101     80.780)
    U4Step::MockOpticksBoundaryIdentity problem step  idx 9918 type 1 U4Step::Name NOT_AT_BOUNDARY cosThetaSign 0 spec  boundary 0 kludge_prim_idx 0
     pre  U4StepPoint::DescPositionTime (    -30.193     -8.788   -990.000      0.000)
     post U4StepPoint::DescPositionTime (    -30.193     -8.788  13908.053     76.415)
    2022-07-11 01:16:39.700 ERROR [12213] [U4Random::check_cursor_vs_tagslot@487]  m_seq_index 9918 cursor 53 slot 52 cursor_slot_match 0
     PROBABLY SOME RANDOM CONSUMPTION LACKS SEvt::AddTag CALLS 
    U4Step::MockOpticksBoundaryIdentity problem step  idx 9917 type 5 U4Step::Name UNEXPECTED cosThetaSign 0 spec LS///LS boundary 4294967295 kludge_prim_idx 0
     pre  U4StepPoint::DescPositionTime (     11.630     12.447   -990.000      0.000)
     post U4StepPoint::DescPositionTime (     11.630     12.447  17824.000     96.501)
    U4Step::MockOpticksBoundaryIdentity problem step  idx 9917 type 5 U4Step::Name UNEXPECTED cosThetaSign 0 spec LS///LS boundary 4294967295 kludge_prim_idx 0
     pre  U4StepPoint::DescPositionTime (     11.630     12.447  17824.000     96.501)
     post U4StepPoint::DescPositionTime (     11.


::

    PIDX=9920 ./U4RecorderTest.sh run
    PIDX=9918 ./U4RecorderTest.sh run
    PIDX=9956 ./U4RecorderTest.sh run




::


    Program received signal SIGSEGV, Segmentation fault.
    0x00007ffff7b96a70 in G4VPhysicalVolume::GetLogicalVolume (this=0x0) at /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/include/Geant4/G4VPhysicalVolume.icc:49
    49	  return flogical;
    Missing separate debuginfos, use: debuginfo-install bzip2-libs-1.0.6-13.el7.x86_64 cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 freetype-2.8-12.el7_6.1.x86_64 glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libICE-1.0.9-9.el7.x86_64 libSM-1.2.2-2.el7.x86_64 libX11-1.6.7-4.el7_9.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-59.el7_9.1.x86_64 libicu-50.2-4.el7_7.x86_64 libidn-1.28-4.el7.x86_64 libpng-1.5.13-7.el7_2.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libuuid-2.23.2-59.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-25.el7_9.x86_64 openssl-libs-1.0.2k-24.el7_9.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007ffff7b96a70 in G4VPhysicalVolume::GetLogicalVolume (this=0x0) at /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/include/Geant4/G4VPhysicalVolume.icc:49
    #1  0x00007ffff7b9ad40 in U4Step::Solid (point=0xa55040) at /data/blyth/junotop/opticks/u4/U4Step.h:436
    #2  0x00007ffff7b9a641 in U4Step::KludgePrimIdx (step=0xa54f10, type=1, idx=8374) at /data/blyth/junotop/opticks/u4/U4Step.h:192
    #3  0x00007ffff7b9a2c8 in U4Step::MockOpticksBoundaryIdentity (current_photon=..., step=0xa54f10, idx=8374) at /data/blyth/junotop/opticks/u4/U4Step.h:104
    #4  0x00007ffff7b9cf61 in U4Recorder::UserSteppingAction_Optical<InstrumentedG4OpBoundaryProcess> (this=0xabafe0, step=0xa54f10) at /data/blyth/junotop/opticks/u4/U4Recorder.cc:257
    #5  0x00007ffff7b9cbed in U4Recorder::UserSteppingAction<InstrumentedG4OpBoundaryProcess> (this=0xabafe0, step=0xa54f10) at /data/blyth/junotop/opticks/u4/U4Recorder.cc:83
    #6  0x0000000000413625 in U4RecorderTest::UserSteppingAction (this=0x7fffffff54b0, step=0xa54f10) at /data/blyth/junotop/opticks/u4/tests/U4RecorderTest.cc:148
    #7  0x00007ffff44ace1d in G4SteppingManager::Stepping() () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #8  0x00007ffff44b8472 in G4TrackingManager::ProcessOneTrack(G4Track*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #9  0x00007ffff46ef389 in G4EventManager::DoProcessing(G4Event*) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #10 0x00007ffff498aa6f in G4RunManager::DoEventLoop(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #11 0x00007ffff498853e in G4RunManager::BeamOn(int, char const*, int) () from /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #12 0x0000000000413b21 in main (argc=1, argv=0x7fffffff6438) at /data/blyth/junotop/opticks/u4/tests/U4RecorderTest.cc:200
    (gdb) 




The untagged consumption looks to be at the end of the history 

::

    u4t ; PIDX=9920 ./U4RecorderTest.sh run 


    2022-07-11 02:52:24.585 INFO  [26174] [U4Random::flat@423]  SEvt::PIDX 9920 m_seq_index 9920 m_seq_nv  256 cursor   48 idx 2539568 d    0.69924
    2022-07-11 02:52:24.585 INFO  [26174] [SEvt::addTag@804]  idx 9920 PIDX 9920 tag 4 flat 0.69924 evt.tag 0x12265820 tagr.slot 48
    2022-07-11 02:52:24.585 INFO  [26174] [U4Random::flat@423]  SEvt::PIDX 9920 m_seq_index 9920 m_seq_nv  256 cursor   49 idx 2539569 d    0.49888
    2022-07-11 02:52:24.585 INFO  [26174] [SEvt::addTag@804]  idx 9920 PIDX 9920 tag 5 flat 0.498883 evt.tag 0x12265820 tagr.slot 49
    ShimG4OpRayleigh::PostStepGetPhysicalInteractionLength PIDX 9920 currentInteractionLength 1000000.0000000 theNumberOfInteractionLengthLeft  0.6953840 value 695383.9375000
    2022-07-11 02:52:24.585 INFO  [26174] [U4Random::flat@423]  SEvt::PIDX 9920 m_seq_index 9920 m_seq_nv  256 cursor   50 idx 2539570 d    0.95529
    2022-07-11 02:52:24.585 INFO  [26174] [SEvt::addTag@804]  idx 9920 PIDX 9920 tag 6 flat 0.955289 evt.tag 0x12265820 tagr.slot 50
    ShimG4OpAbsorption::PostStepGetPhysicalInteractionLength PIDX 9920 currentInteractionLength 1987.1562558 theNumberOfInteractionLengthLeft  0.0457416 value 90.8957291
    2022-07-11 02:52:24.585 INFO  [26174] [U4Random::flat@423]  SEvt::PIDX 9920 m_seq_index 9920 m_seq_nv  256 cursor   51 idx 2539571 d    0.51655
    2022-07-11 02:52:24.585 INFO  [26174] [U4Random::flat@423]  SEvt::PIDX 9920 m_seq_index 9920 m_seq_nv  256 cursor   52 idx 2539572 d    0.98802
    2022-07-11 02:52:24.585 INFO  [26174] [SEvt::addTag@804]  idx 9920 PIDX 9920 tag 9 flat 0.988018 evt.tag 0x12265820 tagr.slot 51
    U4Step::MockOpticksBoundaryIdentity problem step  idx 9920 type 2 U4Step::Name MOTHER_TO_CHILD cosThetaSign -1 spec Pyrex/HamamatsuR12860_PMT_20inch_photocathode_logsurf2/HamamatsuR12860_PMT_20inch_photocathode_logsurf1/Vacuum boundary 32 kludge_prim_idx -1 kludge_prim_idx_ 65535
     pre  U4StepPoint::DescPositionTime (  -9339.653   7475.979  15211.020    107.270)
     post U4StepPoint::DescPositionTime (  -9343.542   7480.095  15216.312    107.309)
    2022-07-11 02:52:24.585 ERROR [26174] [U4Random::check_cursor_vs_tagslot@489]  m_seq_index 9920 cursor 53 slot 52 cursor_slot_match 0
     PROBABLY SOME RANDOM CONSUMPTION LACKS SEvt::AddTag CALLS 
    2022-07-11 02:52:24.585 INFO  [26174] [SEvt::beginPhoton@535]  idx 9919


     post U4StepPoint::DescPositionTime (  -3694.903  -2705.972  18808.381    109.800)
    U4Step::MockOpticksBoundaryIdentity problem step  idx 9918 type 2 U4Step::Name MOTHER_TO_CHILD cosThetaSign -1 spec Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum boundary 35 kludge_prim_idx -1 kludge_prim_idx_ 65535
     pre  U4StepPoint::DescPositionTime (  -3694.903  -2705.972  18808.381    109.800)
     post U4StepPoint::DescPositionTime (  -3699.417  -2711.157  18816.101    109.852)
    2022-07-11 02:52:24.587 ERROR [26174] [U4Random::check_cursor_vs_tagslot@489]  m_seq_index 9918 cursor 53 slot 52 cursor_slot_match 0
     PROBABLY SOME RANDOM CONSUMPTION LACKS SEvt::AddTag CALLS 
    2022-07-11 02:52:24.587 INFO  [26174] [SEvt::beginPhoton@535]  idx 9917



