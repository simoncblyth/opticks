IDENTIFIED CAUSE : G4CXTest_raindrop_shows_Geant4_Process_Reorder_doesnt_fix_velocity_after_reflection_in_Geant4_1120
======================================================================================================================

Overview
----------

After process reorder putting boundary after scintillation(reemission)

* 1042 with or without the kludge gives expected velocities
* 1120 without kludge, expected velocity after transmit but not after reflect 
* 1121 without kludge

  * NOT:WITH_CUSTOM4 using InstrumentedG4OpBoundaryProcess : expected velocity after transmit but not after reflect 
  * NOT:WITH_CUSTOM4 NOT:WITH_INSTRUMENTED_DEBUG corresponds to standard G4OpBoundaryProcess 

The observations below indicate than when switching to Geant4 11.1? 11.2.0 11.2.1 have to use 
newly customized C4OpBoundaryProcess.cc (and InstrumentedG4OpBoundaryProcess) suitable 

The cause of the issue is divergence between the velocity mechanics in the boundary process 
and the rest of Geant4


Observations
-------------


1120::

    [simon@localhost tests]$ ./G4CXTest_raindrop.sh
    ...
    === opticks-setup-geant4- : sourcing /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J24.1.x-g411/ExternalLibs/Geant4/11.2.0/bin/geant4.sh
             BASH_SOURCE : ./G4CXTest_raindrop.sh 
                    GEOM : RaindropRockAirWater 
                 VERSION : 0 
                     TMP : /data/simon/opticks 
                   AFOLD : /data/simon/opticks/GEOM/RaindropRockAirWater/G4CXTest/ALL0/A000 
                   BFOLD : /data/simon/opticks/GEOM/RaindropRockAirWater/G4CXTest/ALL0/B000 
                 evtfold : /data/simon/opticks/GEOM/RaindropRockAirWater 
                     CVD :  
    CUDA_VISIBLE_DEVICES : 1 
                      BP :  
                     arg : info_run_ana 
    2024-04-16 16:13:28.366 INFO  [378086] [G4CXApp::Create@334] U4Recorder::Switches
    NOT:WITH_CUSTOM4
    NOT:WITH_PMTSIM
    NOT:PMTSIM_STANDALONE
    NOT:PRODUCTION


    ab.qcf.aqu
    qcf.aqu : np.c_[n,x,u][o][lim] : uniques in descending count order with first index x
    [[b'49930' b'23001' b'TO BT SA                                                                                        ']
     [b'46609' b'1' b'TO BR BT SA                                                                                     ']
     [b'2474' b'26' b'TO BR BR BT SA                                                                                  ']
     [b'835' b'1570' b'TO BR BR BR BT SA                                                                               ']
     [b'102' b'12323' b'TO BR BR BR BR BT SA                                                                            ']
     [b'39' b'21640' b'TO BR BR BR BR BR BT SA                                                                         ']
     [b'5' b'18141' b'TO BR BR BR BR BR BR BT SA                                                                      ']
     [b'4' b'22738' b'TO BR BR BR BR BR BR BR BT SA                                                                   ']
     [b'1' b'0' b'TO SA                                                                                           ']
     [b'1' b'26998' b'TO BR BR BR BR BR BR BR BR BT                                                                   ']]
    ab.qcf.bqu
    qcf.bqu : np.c_[n,x,u][o][lim] : uniques in descending count order with first index x
    [[b'49905' b'23007' b'TO BT SA                                                                                        ']
     [b'46617' b'0' b'TO BR BT SA                                                                                     ']
     [b'2461' b'26' b'TO BR BR BT SA                                                                                  ']
     [b'819' b'7529' b'TO BR BR BR BT SA                                                                               ']
     [b'144' b'6326' b'TO BR BR BR BR BT SA                                                                            ']
     [b'43' b'12970' b'TO BR BR BR BR BR BT SA                                                                         ']
     [b'9' b'17559' b'TO BR BR BR BR BR BR BT SA                                                                      ']
     [b'2' b'77210' b'TO BR BR BR BR BR BR BR BR BT SA                                                                ']]
    a.CHECK :  
    b.CHECK :  
    ab.qcf[:40]
    QCF qcf :  
    a.q 100000 b.q 100000 lim slice(None, None, None) 
    c2sum :     7.5618 c2n :     6.0000 c2per:     1.2603  C2CUT:   30 
    c2sum/c2n:c2per(C2CUT)   7.56/6:1.260 (30) pv[1.000,> 0.05 : null-hyp ] 

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][0:40]  ## A-B history frequency chi2 comparison 
    [[' 0' 'TO BT SA                        ' ' 0' ' 49930  49905' ' 0.0063' ' 23001  23007']
     [' 1' 'TO BR BT SA                     ' ' 1' ' 46609  46617' ' 0.0007' '     1      0']
     [' 2' 'TO BR BR BT SA                  ' ' 2' '  2474   2461' ' 0.0342' '    26     26']
     [' 3' 'TO BR BR BR BT SA               ' ' 3' '   835    819' ' 0.1548' '  1570   7529']
     [' 4' 'TO BR BR BR BR BT SA            ' ' 4' '   102    144' ' 7.1707' ' 12323   6326']
     [' 5' 'TO BR BR BR BR BR BT SA         ' ' 5' '    39     43' ' 0.1951' ' 21640  12970']
     [' 6' 'TO BR BR BR BR BR BR BT SA      ' ' 6' '     5      9' ' 0.0000' ' 18141  17559']
     [' 7' 'TO BR BR BR BR BR BR BR BT SA   ' ' 7' '     4      0' ' 0.0000' ' 22738     -1']
     [' 8' 'TO BR BR BR BR BR BR BR BR BT SA' ' 8' '     0      2' ' 0.0000' '    -1  77210']
     [' 9' 'TO SA                           ' ' 9' '     1      0' ' 0.0000' '     0     -1']
     ['10' 'TO BR BR BR BR BR BR BR BR BT   ' '10' '     1      0' ' 0.0000' ' 26998     -1']]

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][bzero]  ## in A but not B 
    []

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][azero]  ## in B but not A 
    []
    ]----- repr(ab) 
    np.c_[np.unique(a.q, return_counts=True)] 
    [[b'TO BR BR BR BR BR BR BR BR BT                                                                   ' b'1']
     [b'TO BR BR BR BR BR BR BR BT SA                                                                   ' b'4']
     [b'TO BR BR BR BR BR BR BT SA                                                                      ' b'5']
     [b'TO BR BR BR BR BR BT SA                                                                         ' b'39']
     [b'TO BR BR BR BR BT SA                                                                            ' b'102']
     [b'TO BR BR BR BT SA                                                                               ' b'835']
     [b'TO BR BR BT SA                                                                                  ' b'2474']
     [b'TO BR BT SA                                                                                     ' b'46609']
     [b'TO BT SA                                                                                        ' b'49930']
     [b'TO SA                                                                                           ' b'1']]
    np.c_[np.unique(b.q, return_counts=True)] 
    [[b'TO BR BR BR BR BR BR BR BR BT SA                                                                ' b'2']
     [b'TO BR BR BR BR BR BR BT SA                                                                      ' b'9']
     [b'TO BR BR BR BR BR BT SA                                                                         ' b'43']
     [b'TO BR BR BR BR BT SA                                                                            ' b'144']
     [b'TO BR BR BR BT SA                                                                               ' b'819']
     [b'TO BR BR BT SA                                                                                  ' b'2461']
     [b'TO BR BT SA                                                                                     ' b'46617']
     [b'TO BT SA                                                                                        ' b'49905']]
    PICK=B MODE=3 SELECT="TO BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    REC=/data/simon/opticks/GEOM/RaindropRockAirWater/G4CXTest/ALL0/B000/TO_BT_SA ~/o/examples/UseGeometryShader/run.sh
    speed len/min/max for : 0 -> 1 : TO -> BT :    49905 224.901 224.901 
    speed len/min/max for : 1 -> 2 : BT -> SA :    49905 299.792 299.793 
    e.f.NPFold_meta.U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE:0 
    e.f.NPFold_meta.G4VERSION_NUMBER:1120 
    _pos.shape (49905, 3) 
    _beg.shape (49905, 3) 
    _poi.shape (49905, 3, 3) 
    PICK=B MODE=3 SELECT="TO BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    REC=/data/simon/opticks/GEOM/RaindropRockAirWater/G4CXTest/ALL0/B000/TO_BR_BT_SA ~/o/examples/UseGeometryShader/run.sh
    speed len/min/max for : 0 -> 1 : TO -> BR :    46617 224.901 224.901 
    speed len/min/max for : 1 -> 2 : BR -> BT :    46617 299.792 299.793 
    speed len/min/max for : 2 -> 3 : BT -> SA :    46617 299.792 299.793 
    e.f.NPFold_meta.U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE:0 
    e.f.NPFold_meta.G4VERSION_NUMBER:1120 
    _pos.shape (46617, 3) 
    _beg.shape (46617, 3) 
    _poi.shape (46617, 4, 3) 

    In [1]: 


1120 with UseGivenVelocity_KLUDGE regains expected velocities::

    PICK=B MODE=3 SELECT="TO BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    REC=/data/simon/opticks/GEOM/RaindropRockAirWater/G4CXTest/ALL1/B000/TO_BT_SA ~/o/examples/UseGeometryShader/run.sh
    speed len/min/max for : 0 -> 1 : TO -> BT :    49905 224.901 224.901 
    speed len/min/max for : 1 -> 2 : BT -> SA :    49905 299.792 299.793 
    e.f.NPFold_meta.U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE:1 
    e.f.NPFold_meta.G4VERSION_NUMBER:1120 
    _pos.shape (49905, 3) 
    _beg.shape (49905, 3) 
    _poi.shape (49905, 3, 3) 
    PICK=B MODE=3 SELECT="TO BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    REC=/data/simon/opticks/GEOM/RaindropRockAirWater/G4CXTest/ALL1/B000/TO_BR_BT_SA ~/o/examples/UseGeometryShader/run.sh
    speed len/min/max for : 0 -> 1 : TO -> BR :    46617 224.901 224.901 
    speed len/min/max for : 1 -> 2 : BR -> BT :    46617 224.901 224.901 
    speed len/min/max for : 2 -> 3 : BT -> SA :    46617 299.792 299.793 
    e.f.NPFold_meta.U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE:1 
    e.f.NPFold_meta.G4VERSION_NUMBER:1120 
    _pos.shape (46617, 3) 
    _beg.shape (46617, 3) 
    _poi.shape (46617, 4, 3) 


1121 debug build of CLHEP,Geant4,Opticks NOT:WITH_CUSTOM4 no kludge : wrong velocity after BR::

    [blyth@localhost u4]$ ~/o/g4cx/tests/G4CXTest_raindrop.sh
    ...
    2024-04-23 19:17:56.943 INFO  [358025] [G4CXApp::G4CXApp@160] 
    U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE:0
    $Name: geant4-11-02-patch-01 [MT]$ (16-February-2024)U4Recorder::Switches
    NOT:WITH_CUSTOM4
    NOT:WITH_PMTSIM
    NOT:PMTSIM_STANDALONE
    NOT:PRODUCTION
    ...
    PICK=B MODE=3 SELECT="TO BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    REC=/home/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0/B000/TO_BT_SA ~/o/examples/UseGeometryShader/run.sh
    speed len/min/max for : 0 -> 1 : TO -> BT :    49905 224.901 224.901 
    speed len/min/max for : 1 -> 2 : BT -> SA :    49905 299.792 299.793 
    e.f.NPFold_meta.U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE:0 
    e.f.NPFold_meta.G4VERSION_NUMBER:1121 
    _pos.shape (49905, 3) 
    _beg.shape (49905, 3) 
    _poi.shape (49905, 3, 3) 
    PICK=B MODE=3 SELECT="TO BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    REC=/home/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0/B000/TO_BR_BT_SA ~/o/examples/UseGeometryShader/run.sh
    speed len/min/max for : 0 -> 1 : TO -> BR :    46617 224.901 224.901 
    speed len/min/max for : 1 -> 2 : BR -> BT :    46617 299.792 299.793 
    speed len/min/max for : 2 -> 3 : BT -> SA :    46617 299.792 299.793 
    e.f.NPFold_meta.U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE:0 
    e.f.NPFold_meta.G4VERSION_NUMBER:1121 

    BP=InstrumentedG4OpBoundaryProcess::PostStepDoIt ~/o/g4cx/tests/G4CXTest_raindrop.sh
    BP=G4ParticleChange::ProposeVelocity ~/o/g4cx/tests/G4CXTest_raindrop.sh

HMM: this could be incompatibility between the InstrumentedG4OpBoundaryProcess which is
based on the 1042 version and the rest of 1121::

    (gdb) bt
    #0  G4ParticleChange::ProposeVelocity (this=0x2a8d7a8, finalVelocity=224.90056864216055) at /data/blyth/opticks_externals/geant4_1121/include/Geant4/G4ParticleChange.icc:50
    #1  0x00007ffff7d40b86 in InstrumentedG4OpBoundaryProcess::PostStepDoIt_ (this=0x2a8d790, aTrack=..., aStep=...) at /home/blyth/opticks/u4/InstrumentedG4OpBoundaryProcess.cc:438
    #2  0x00007ffff7d40a9f in InstrumentedG4OpBoundaryProcess::PostStepDoIt (this=0x2a8d790, aTrack=..., aStep=...) at /home/blyth/opticks/u4/InstrumentedG4OpBoundaryProcess.cc:323
    #3  0x00007ffff6d864a6 in G4SteppingManager::InvokePSDIP (this=this@entry=0x7231b0, np=np@entry=3)
        at /data/blyth/opticks_externals/geant4_1121.build/geant4-v11.2.1/source/tracking/src/G4SteppingManager.cc:818
    #4  0x00007ffff6d865d5 in G4SteppingManager::InvokePostStepDoItProcs (this=this@entry=0x7231b0)
        at /data/blyth/opticks_externals/geant4_1121.build/geant4-v11.2.1/source/tracking/src/G4SteppingManager.cc:790
    #5  0x00007ffff6d86a65 in G4SteppingManager::Stepping (this=0x7231b0) at /data/blyth/opticks_externals/geant4_1121.build/geant4-v11.2.1/source/tracking/src/G4SteppingManager.cc:225
    #6  0x00007ffff6d9a029 in G4TrackingManager::ProcessOneTrack (this=0x720670, apValueG4Track=apValueG4Track@entry=0x110e630)
        at /data/blyth/opticks_externals/geant4_1121.build/geant4-v11.2.1/source/tracking/src/G4TrackingManager.cc:133
    #7  0x00007ffff6dda51d in G4EventManager::DoProcessing (this=0x721ad0, anEvent=<optimized out>) at /data/blyth/opticks_externals/geant4_1121.build/geant4-v11.2.1/source/event/src/G4EventManager.cc:211
    #8  0x00007ffff6ddab3a in G4EventManager::ProcessOneEvent (this=<optimized out>, anEvent=<optimized out>)
        at /data/blyth/opticks_externals/geant4_1121.build/geant4-v11.2.1/source/event/src/G4EventManager.cc:447
    #9  0x00007ffff6e971a9 in G4RunManager::ProcessOneEvent (this=0x721880, i_event=0) at /data/blyth/opticks_externals/geant4_1121.build/geant4-v11.2.1/source/run/src/G4RunManager.cc:404
    #10 0x00007ffff6e95dfc in G4RunManager::DoEventLoop (this=0x721880, n_event=1, macroFile=<optimized out>, n_select=<optimized out>)
        at /data/blyth/opticks_externals/geant4_1121.build/geant4-v11.2.1/source/run/src/G4RunManager.cc:370
    #11 0x00007ffff6e95db6 in G4RunManager::BeamOn (this=0x721880, n_event=1, macroFile=0x0, n_select=-1) at /data/blyth/opticks_externals/geant4_1121.build/geant4-v11.2.1/source/run/src/G4RunManager.cc:261
    #12 0x000000000040a9ad in G4CXApp::BeamOn (this=0x788810) at /home/blyth/opticks/g4cx/tests/G4CXApp.h:344
    #13 0x000000000040aab9 in G4CXApp::Main () at /home/blyth/opticks/g4cx/tests/G4CXApp.h:351
    #14 0x000000000040ac47 in main (argc=1, argv=0x7fffffffc0d8) at /home/blyth/opticks/g4cx/tests/G4CXTest.cc:13
    (gdb) 



::
 
     117 #include "U4UniformRand.h"
     118 NP* U4UniformRand::UU = nullptr ;
     119 // UU gets set by U4Recorder::saveOrLoadStates when doing single photon reruns



YEP: Changing to standard G4OpBoundaryProcess avoids issue
--------------------------------------------------------------

::

    2024-04-23 20:30:22.427 INFO  [33315] [G4CXApp::G4CXApp@160] 
    U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE:0
    $Name: geant4-11-02-patch-01 [MT]$ (16-February-2024)U4Recorder::Switches
    NOT:WITH_CUSTOM4
    NOT:WITH_PMTSIM
    NOT:PMTSIM_STANDALONE
    NOT:PRODUCTION
    NOT:WITH_INSTRUMENTED_DEBUG

    ...

    PICK=B MODE=3 SELECT="TO BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    REC=/home/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0/B000/TO_BT_SA ~/o/examples/UseGeometryShader/run.sh
    speed len/min/max for : 0 -> 1 : TO -> BT :    49905 224.901 224.901 
    speed len/min/max for : 1 -> 2 : BT -> SA :    49905 299.792 299.793 
    e.f.NPFold_meta.U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE:0 
    e.f.NPFold_meta.G4VERSION_NUMBER:1121 
    _pos.shape (49905, 3) 
    _beg.shape (49905, 3) 
    _poi.shape (49905, 3, 3) 
    PICK=B MODE=3 SELECT="TO BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    REC=/home/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0/B000/TO_BR_BT_SA ~/o/examples/UseGeometryShader/run.sh
    speed len/min/max for : 0 -> 1 : TO -> BR :    46617 224.901 224.901 
    speed len/min/max for : 1 -> 2 : BR -> BT :    46617 224.901 224.901 
    speed len/min/max for : 2 -> 3 : BT -> SA :    46617 299.792 299.793 
    e.f.NPFold_meta.U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE:0 
    e.f.NPFold_meta.G4VERSION_NUMBER:1121 
    _pos.shape (46617, 3) 
    _beg.shape (46617, 3) 
    _poi.shape (46617, 4, 3) 


