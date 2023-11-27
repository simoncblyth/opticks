G4CXTest_GEOM_num_photon_assert.rst
======================================

Are genstep added twice ? 


~/opticks/g4cx/tests/G4CXTest_GEOM.sh::

     62 num=100000
     63 NUM=${NUM:-$num}
     64 
     65 export OPTICKS_MAX_PHOTON=100000
     66 
     67 
     68 #srm=SRM_DEFAULT
     69 srm=SRM_TORCH
     70 #srm=SRM_INPUT_PHOTON
     71 #srm=SRM_INPUT_GENSTEP    ## NOT IMPLEMENTED FOR GEANT4
     72 #srm=SRM_GUN
     73 export OPTICKS_RUNNING_MODE=$srm
     74 
     75 echo $BASH_SOURCE OPTICKS_RUNNING_MODE $OPTICKS_RUNNING_MODE
     76 
     77 if [ "$OPTICKS_RUNNING_MODE" == "SRM_TORCH" ]; then
     78     export SEvent_MakeGensteps_num_ph=$NUM
     79     #src="rectangle"
     80     #src="disc"
     81     src="sphere"
     82 


::

    N[blyth@localhost opticks]$ ~/opticks/g4cx/tests/G4CXTest_GEOM.sh
    /home/blyth/opticks/g4cx/tests/G4CXTest_GEOM.sh OPTICKS_RUNNING_MODE SRM_TORCH
                                           BASH_SOURCE : /home/blyth/opticks/g4cx/tests/G4CXTest_GEOM.sh 
                                                  SDIR : /home/blyth/opticks/g4cx/tests 



    2023-11-27 16:42:56.800 INFO  [337434] [G4CXApp::BeamOn@325] [ OPTICKS_NUM_EVENT=3
    2023-11-27 16:44:28.173 INFO  [337434] [U4Recorder::BeginOfRunAction@253] 
    2023-11-27 16:44:28.173 INFO  [337434] [G4CXApp::GeneratePrimaries@218] [ SEventConfig::RunningModeLabel SRM_TORCH
    2023-11-27 16:44:28.174 INFO  [337434] [SEvent::MakeGensteps@121] num_ph 100000 dump 0
    U4VPrimaryGenerator::GeneratePrimaries ph (100000, 4, 4, )
    2023-11-27 16:44:28.264 INFO  [337434] [G4CXApp::GeneratePrimaries@243] ]
    2023-11-27 16:44:28.309 INFO  [337434] [SEvt::hostside_running_resize_@2146] resizing photon 0 to evt.num_photon 100000
    2023-11-27 16:44:35.748 INFO  [337434] [U4Recorder::PreUserTrackingAction_Optical@399]  modulo 100000 : ulabel.id 0
    2023-11-27 16:44:35.748 INFO  [337434] [SEvent::MakeGensteps@121] num_ph 100000 dump 0
    2023-11-27 16:44:36.921 INFO  [337434] [SEvt::save@3732]  dir /home/blyth/tmp/GEOM/J23_1_0_rc3_ok0/G4CXTest/ALL0/p001 index 1 instance 0 OPTICKS_SAVE_COMP  genstep,photon,record,seq,prd,hit,domain,inphoton,tag,flat,aux,sup
    2023-11-27 16:44:39.075 INFO  [337434] [G4CXApp::GeneratePrimaries@218] [ SEventConfig::RunningModeLabel SRM_TORCH
    2023-11-27 16:44:39.075 INFO  [337434] [SEvent::MakeGensteps@121] num_ph 100000 dump 0
    2023-11-27 16:44:39.075 FATAL [337434] [SEvt::setNumPhoton@2028]  num_photon 200000 evt.max_photon 100000
    G4CXTest: /home/blyth/junotop/opticks/sysrap/SEvt.cc:2029: void SEvt::setNumPhoton(unsigned int): Assertion `num_photon_allowed' failed.
    /home/blyth/opticks/g4cx/tests/G4CXTest_GEOM.sh: line 162: 337434 Aborted                 (core dumped) $bin
    /home/blyth/opticks/g4cx/tests/G4CXTest_GEOM.sh : run error
    N[blyth@localhost opticks]$ 
    N[blyth@localhost opticks]$ 


::

    2023-11-27 16:51:54.246 INFO  [351326] [G4CXApp::BeamOn@325] [ OPTICKS_NUM_EVENT=3
    2023-11-27 16:53:24.361 INFO  [351326] [U4Recorder::BeginOfRunAction@253] 
    2023-11-27 16:53:24.361 INFO  [351326] [G4CXApp::GeneratePrimaries@218] [ SEventConfig::RunningModeLabel SRM_TORCH
    2023-11-27 16:53:24.361 INFO  [351326] [SEvent::MakeGensteps@121] num_ph 100000 dump 0
    U4VPrimaryGenerator::GeneratePrimaries ph (100000, 4, 4, )
    2023-11-27 16:53:24.453 INFO  [351326] [G4CXApp::GeneratePrimaries@243] ]
    2023-11-27 16:53:24.497 INFO  [351326] [SEvt::hostside_running_resize_@2146] resizing photon 0 to evt.num_photon 100000
    2023-11-27 16:53:31.852 INFO  [351326] [U4Recorder::PreUserTrackingAction_Optical@399]  modulo 100000 : ulabel.id 0
    2023-11-27 16:53:31.853 INFO  [351326] [SEvent::MakeGensteps@121] num_ph 100000 dump 0
    2023-11-27 16:53:33.016 INFO  [351326] [SEvt::save@3732]  dir /home/blyth/tmp/GEOM/J23_1_0_rc3_ok0/G4CXTest/ALL0/p001 index 1 instance 0 OPTICKS_SAVE_COMP  genstep,photon,record,seq,prd,hit,domain,inphoton,tag,flat,aux,sup
    2023-11-27 16:53:34.746 INFO  [351326] [G4CXApp::GeneratePrimaries@218] [ SEventConfig::RunningModeLabel SRM_TORCH
    2023-11-27 16:53:34.746 INFO  [351326] [SEvent::MakeGensteps@121] num_ph 100000 dump 0
    2023-11-27 16:53:34.746 FATAL [351326] [SEvt::setNumPhoton@2028]  num_photon 200000 evt.max_photon 100000
    G4CXTest: /home/blyth/junotop/opticks/sysrap/SEvt.cc:2029: void SEvt::setNumPhoton(unsigned int): Assertion `num_photon_allowed' failed.

    Thread 1 "G4CXTest" received signal SIGABRT, Aborted.
    0x00007ffff2756387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff2756387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff2757a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff274f1a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff274f252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff3e412bc in SEvt::setNumPhoton (this=0x6c5e30, num_photon=200000) at /home/blyth/junotop/opticks/sysrap/SEvt.cc:2029
    #5  0x00007ffff3e4101c in SEvt::addGenstep (this=0x6c5e30, q_=...) at /home/blyth/junotop/opticks/sysrap/SEvt.cc:1995
    #6  0x00007ffff3e401ef in SEvt::addGenstep (this=0x6c5e30, a=0xee55de0) at /home/blyth/junotop/opticks/sysrap/SEvt.cc:1874
    #7  0x00007ffff3e3dba4 in SEvt::addTorchGenstep (this=0x6c5e30) at /home/blyth/junotop/opticks/sysrap/SEvt.cc:1227
    #8  0x000000000040a06c in G4CXApp::GeneratePrimaries (this=0x6c5b90, event=0x13282690) at /home/blyth/junotop/opticks/g4cx/tests/G4CXApp.h:230
    #9  0x00007ffff70d6d7a in G4RunManager::GenerateEvent(int) ()
       from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #10 0x00007ffff70d4a8c in G4RunManager::DoEventLoop(int, char const*, int) ()
       from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #11 0x00007ffff70d24de in G4RunManager::BeamOn(int, char const*, int) ()
       from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #12 0x000000000040a907 in G4CXApp::BeamOn (this=0x6c5b90) at /home/blyth/junotop/opticks/g4cx/tests/G4CXApp.h:326
    #13 0x000000000040aa13 in G4CXApp::Main () at /home/blyth/junotop/opticks/g4cx/tests/G4CXApp.h:333
    #14 0x000000000040ab9d in main (argc=1, argv=0x7fffffff1eb8) at /home/blyth/junotop/opticks/g4cx/tests/G4CXTest.cc:16


::

    226     else if(SEventConfig::IsRunningModeTorch())
    227     {
    228         SEvt* sev = SEvt::Get_ECPU();
    229         assert(sev);
    230         sev->addTorchGenstep();
    231         // formerly added to both instances via static SEvt::AddTorchGenstep()
    232         U4VPrimaryGenerator::GeneratePrimaries(event);
    233     }


HMM: getting the torch gensteps into Geant4 means have to have them earlier. 

Maybe could avoid SEvt at this early stage, just get GeneratePrimaries
to use SGenerate::GeneratePhotons ?


::

     668 void SEvt::addFrameGenstep()
     669 {

     712             else if( has_torch )
     713             {
     714                 if(isEGPU())
     715                 {
     716                     assertZeroGensteps();
     717                     // just filling the storch struct from config (no generation yet)
     718                     // so repeating for CPU and GPU instances is no problem 
     719                     NP* togs = SEvent::MakeTorchGensteps();
     720                     addGenstep(togs);
     721                 }
     722             }




::

    084 /**
     85 SGenerate::GeneratePhotons
     86 ----------------------------
     87 
     88 Does high level genstep handling, prepares MOCK CURAND, 
     89 creates seeds, creates photon array. 
     90 The details of the generation are done by storch::generate or scarrier:generate
     91 
     92 **/
     93 
     94 inline NP* SGenerate::GeneratePhotons(const NP* gs_)
     95 {


    126 inline void U4VPrimaryGenerator::GeneratePrimaries(G4Event* event)
    127 {
    128     int idx = 1 ; // SEvt::ECPU 
    129     NP* ph = SGenerate::GeneratePhotons(idx);
    130     // TODO: these *ph* are effectively input photons (even though generated from gensteps),
    131     //       should associate as such in the SEvt to retain access to these
    132     //
    133 



::

    2023-11-27 19:12:13.272 INFO  [115122] [G4CXApp::BeamOn@331] [ OPTICKS_NUM_EVENT=3
    2023-11-27 19:13:43.063 INFO  [115122] [U4Recorder::BeginOfRunAction@253] 
    2023-11-27 19:13:43.063 INFO  [115122] [G4CXApp::GeneratePrimaries@223] [ SEventConfig::RunningModeLabel SRM_TORCH eventID 0
    2023-11-27 19:13:43.063 INFO  [115122] [SEvent::MakeGensteps@121] num_ph 100000 dump 0
    U4VPrimaryGenerator::GeneratePrimaries ph (100000, 4, 4, )
    2023-11-27 19:13:43.134 INFO  [115122] [G4CXApp::GeneratePrimaries@249] ]  eventID 0
    2023-11-27 19:13:43.178 INFO  [115122] [SEvt::hostside_running_resize_@2146] resizing photon 0 to evt.num_photon 0
    2023-11-27 19:13:43.178 ERROR [115122] [SEvt::beginPhoton@2274]  not in_range  idx 99999 pho.size  0 label spho (gs:ix:id:gn   09999999999[  0,  0,  0,  0])
    SEvt::beginPhoton FATAL not in_range idx 99999 pho.size  0 label spho (gs:ix:id:gn   09999999999[  0,  0,  0,  0])
    G4CXTest: /home/blyth/junotop/opticks/sysrap/SEvt.cc:2289: void SEvt::beginPhoton(const spho&): Assertion `in_range' failed.
    /home/blyth/opticks/g4cx/tests/G4CXTest_GEOM.sh: line 170: 115122 Aborted                 (core dumped) $bin
    /home/blyth/opticks/g4cx/tests/G4CXTest_GEOM.sh : run error
    N[blyth@localhost opticks]$ 


