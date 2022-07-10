U4Stack_Linux_Darwin_difference
==================================

Linux SBacktrace::Summary missing crucial line with "DsG4Scintillation::ResetNumberOfInteractionLengthLeft"
But scintillation is the only G4VRestDiscreteProcess in play so do not need to Shim.


::

    class DsG4Scintillation : public G4VRestDiscreteProcess, public G4UImessenger

    class G4OpAbsorption : public G4VDiscreteProcess

    class G4OpRayleigh : public G4VDiscreteProcess

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



The deficient SBacktrace::Summary might arise of inline imps.






