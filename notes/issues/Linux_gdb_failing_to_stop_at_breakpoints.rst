Linux_gdb_failing_to_stop_at_breakpoints
==========================================

Overview
---------

Need Debug builds of the Geant4 versions to make progress with this.


blyth WITH_CUSTOM4 1042 works
-------------------------------

But the cvmfs Geant4 is only Release build, so no debug symbols.

::

    [blyth@localhost tests]$ BP=C4OpBoundaryProcess::PostStepDoIt ./G4CXTest_raindrop.sh

    Reading symbols from G4CXTest...
    Function "C4OpBoundaryProcess::PostStepDoIt" not defined.
    Breakpoint 1 (C4OpBoundaryProcess::PostStepDoIt) pending.
    Num     Type           Disp Enb Address    What
    1       breakpoint     keep y   <PENDING>  C4OpBoundaryProcess::PostStepDoIt
    Starting program: /data/blyth/junotop/ExternalLibs/opticks/head/lib/G4CXTest 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    warning: File "/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/lib64/libstdc++.so.6.0.29-gdb.py" auto-loading has been declined by your `auto-load safe-path' set to "$debugdir:$datadir/auto-load".
    To enable execution of this file add
        add-auto-load-safe-path /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/lib64/libstdc++.so.6.0.29-gdb.py
    line to your configuration file "/home/blyth/.gdbinit".
    To completely disable this security protection add
        set auto-load safe-path /
    line to your configuration file "/home/blyth/.gdbinit".
    For more information about this security protection see the
    "Auto-loading safe path" section in the GDB manual.  E.g., run from the shell:
        info "(gdb)Auto-loading safe path"
    2024-04-22 18:38:45.207 INFO  [219952] [G4CXApp::Create@334] U4Recorder::Switches
    WITH_CUSTOM4
    NOT:WITH_PMTSIM
    NOT:PMTSIM_STANDALONE
    NOT:PRODUCTION

    ...

    d 1 "G4CXTest" hit Breakpoint 1, 0x00007ffff4b7cef0 in C4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&) () from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.0-rc3.dc1/ExternalLibs/custom4/0.1.8/lib64/libCustom4.so
    (gdb) bt
    #0  0x00007ffff4b7cef0 in C4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&) ()
       from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.0-rc3.dc1/ExternalLibs/custom4/0.1.8/lib64/libCustom4.so
    #1  0x00007ffff702d8d9 in G4SteppingManager::InvokePSDIP(unsigned long) ()
       from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #2  0x00007ffff702dccb in G4SteppingManager::InvokePostStepDoItProcs() ()
       from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #3  0x00007ffff702b53e in G4SteppingManager::Stepping() ()
       from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #4  0x00007ffff7036aaf in G4TrackingManager::ProcessOneTrack(G4Track*) ()
       from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4tracking.so
    #5  0x00007ffff7071d0d in G4EventManager::DoProcessing(G4Event*) ()
       from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4event.so
    #6  0x00007ffff7111a9f in G4RunManager::DoEventLoop(int, char const*, int) ()
       from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #7  0x00007ffff710f4de in G4RunManager::BeamOn(int, char const*, int) ()
       from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #8  0x000000000040a9d5 in G4CXApp::BeamOn (this=0x6cbca0) at /home/blyth/junotop/opticks/g4cx/tests/G4CXApp.h:344
    #9  0x000000000040aae1 in G4CXApp::Main () at /home/blyth/junotop/opticks/g4cx/tests/G4CXApp.h:351
    #10 0x000000000040ac6f in main (argc=1, argv=0x7fffffff1a08) at /home/blyth/junotop/opticks/g4cx/tests/G4CXTest.cc:13
    (gdb) 





simon NOT-WITH_CUSTOM4 1120 failing to stop at breakpoints
--------------------------------------------------------------


::

    [simon@localhost tests]$ BP=G4OpBoundaryProcess::PostStepDoIt ./G4CXTest_raindrop.sh

    gdb -ex "set breakpoint pending on" -ex "break G4OpBoundaryProcess::PostStepDoIt" -ex "info break" -ex r --args G4CXTest
    Mon Apr 22 18:35:10 CST 2024
    GNU gdb (GDB) 12.1
    Copyright (C) 2022 Free Software Foundation, Inc.
    License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
    This is free software: you are free to change and redistribute it.
    There is NO WARRANTY, to the extent permitted by law.
    Type "show copying" and "show warranty" for details.
    This GDB was configured as "x86_64-pc-linux-gnu".
    Type "show configuration" for configuration details.
    For bug reporting instructions, please see:
    <https://www.gnu.org/software/gdb/bugs/>.
    Find the GDB manual and other documentation resources online at:
        <http://www.gnu.org/software/gdb/documentation/>.

    For help, type "help".
    Type "apropos word" to search for commands related to "word"...
    Reading symbols from G4CXTest...
    Function "G4OpBoundaryProcess::PostStepDoIt" not defined.
    Breakpoint 1 (G4OpBoundaryProcess::PostStepDoIt) pending.
    Num     Type           Disp Enb Address    What
    1       breakpoint     keep y   <PENDING>  G4OpBoundaryProcess::PostStepDoIt
    Starting program: /data/simon/local/opticks_Debug/lib/G4CXTest 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".

    warning: File "/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/lib64/libstdc++.so.6.0.29-gdb.py" auto-loading has been declined by your `auto-load safe-path' set to "$debugdir:$datadir/auto-load".
    To enable execution of this file add
        add-auto-load-safe-path /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/lib64/libstdc++.so.6.0.29-gdb.py
    line to your configuration file "/home/simon/.config/gdb/gdbinit".
    To completely disable this security protection add
        set auto-load safe-path /
    line to your configuration file "/home/simon/.config/gdb/gdbinit".
    For more information about this security protection see the
    "Auto-loading safe path" section in the GDB manual.  E.g., run from the shell:
        info "(gdb)Auto-loading safe path"


    $Name: geant4-11-02 [MT]$ (8-December-2023)U4Recorder::Switches
    NOT:WITH_CUSTOM4
    NOT:WITH_PMTSIM
    NOT:PMTSIM_STANDALONE
    NOT:PRODUCTION


    2024-04-22 18:35:15.750 INFO  [214243] [G4CXApp::BeamOn@343] [ OPTICKS_NUM_EVENT=1
    2024-04-22 18:35:15.864 INFO  [214243] [G4CXApp::GeneratePrimaries@223] [ SEventConfig::RunningModeLabel SRM_TORCH eventID 0
    SGenerate::GeneratePhotons SGenerate__GeneratePhotons_RNG_PRECOOKED : NO 
    U4VPrimaryGenerator::GeneratePrimaries_From_Photons ph (100000, 4, 4, )
     U4VPrimaryGenerator__GeneratePrimaries_From_Photons_DEBUG_GENIDX : 10000 (when +ve, only generate tht photon idx)
    2024-04-22 18:35:15.894 INFO  [214243] [G4CXApp::GeneratePrimaries@253] ]  eventID 0
    2024-04-22 18:35:15.894 INFO  [214243] [U4Recorder::BeginOfEventAction_@309]  eventID 0
    2024-04-22 18:35:15.894 INFO  [214243] [U4Recorder::PreUserTrackingAction_Optical@416]  modulo 100000 : ulabel.id 0
    2024-04-22 18:35:17.138 INFO  [214243] [G4CXApp::BeamOn@345] ]
    [Thread 0x7fffe0d89700 (LWP 214354) exited]
    [Thread 0x7fffe174e700 (LWP 214344) exited]
    [Thread 0x7fffea960000 (LWP 214243) exited]
    [Thread 0x7fffe1f4f700 (LWP 214342) exited]
    [New process 214243]
    [Inferior 1 (process 214243) exited normally]
    (gdb) 
    quit
    Mon Apr 22 18:35:19 CST 2024


Doing what gdb asks avoids the noise but does not get it to stop at the breakpoint::

    [simon@localhost ~]$ cat /home/simon/.config/gdb/gdbinit
    add-auto-load-safe-path /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/lib64/libstdc++.so.6.0.29-gdb.py
    [simon@localhost ~]$ 



