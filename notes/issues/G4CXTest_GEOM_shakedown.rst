G4CXTest_GEOM_shakedown using G4CXTest.cc : standalone bi-simulation starting from GDML
==========================================================================================


FIXED : Issue 1 : GDML resolution
------------------------------------

::

    P[blyth@localhost tests]$ ~/o/G4CXTest_GEOM.sh

    ...

    2024-11-11 17:11:42.960 INFO  [281525] [G4CXApp::Construct@165] [
    U4VolumeMaker::PV name J_2024aug27
    2024-11-11 17:11:42.960 FATAL [281525] [SOpticksResource::GDMLPath@452]  TODO: ELIMINATE THIS : INSTEAD USE GDMLPathFromGEOM 
    U4VolumeMaker::PVG_ name J_2024aug27 gdmlpath - sub - exists 0
    2024-11-11 17:11:42.961 INFO  [281525] [U4VolumeMaker::PVP_@246]  not-WITH_PMTSIM name [J_2024aug27]
    2024-11-11 17:11:42.961 ERROR [281525] [U4SolidMaker::Make@171]  Failed to create solid for qname J_2024aug27 CHECK U4SolidMaker::Make 
    2024-11-11 17:11:42.961 ERROR [281525] [U4VolumeMaker::LV@322]  failed to access solid for name J_2024aug27
    2024-11-11 17:11:42.961 ERROR [281525] [U4VolumeMaker::PV1_@284]  failed to access lv for name J_2024aug27
    2024-11-11 17:11:42.961 ERROR [281525] [U4VolumeMaker::PV@105] returning nullptr for name [J_2024aug27]
    2024-11-11 17:11:42.961 FATAL [281525] [G4CXApp::Construct@167]  FAILED TO CREATE PV : CHECK GEOM envvar 
    U4VolumeMaker::Desc GEOM J_2024aug27 METH PV1_ not-WITH_PMTSIM 
    /home/blyth/o/G4CXTest_GEOM.sh run error
    P[blyth@localhost tests]$ 


::

     01 /**
      2 G4CXTest.cc : Standalone bi-simulation
      3 ======================================
      4 
      5 **/
      6 
      7 #include "OPTICKS_LOG.hh"
      8 #include "G4CXApp.h"
      9 
     10 int main(int argc, char** argv)
     11 {
     12     OPTICKS_LOG(argc, argv);
     13     return G4CXApp::Main();
     14 }
     15 

::

    P[blyth@localhost tests]$ BP=SOpticksResource::GDMLPath ~/o/G4CXTest_GEOM.sh


    (gdb) 
    (gdb) bt
    #0  0x00007ffff7c86e00 in SOpticksResource::GDMLPath(char const*)@plt () from /data/blyth/opticks_Debug/lib/../lib64/libU4.so
    #1  0x00007ffff7cf8e35 in U4VolumeMaker::PVG_ (name=0x7fffffff759f "J_2024aug27") at /home/blyth/opticks/u4/U4VolumeMaker.cc:139
    #2  0x00007ffff7cf8b89 in U4VolumeMaker::PV (name=0x7fffffff759f "J_2024aug27") at /home/blyth/opticks/u4/U4VolumeMaker.cc:101
    #3  0x00007ffff7cf8a26 in U4VolumeMaker::PV () at /home/blyth/opticks/u4/U4VolumeMaker.cc:94
    #4  0x0000000000409a9f in G4CXApp::Construct (this=0x6c31e0) at /home/blyth/opticks/g4cx/tests/G4CXApp.h:166
    #5  0x00007ffff706b95e in G4RunManager::InitializeGeometry() () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #6  0x00007ffff706bb2c in G4RunManager::Initialize() () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #7  0x0000000000409831 in G4CXApp::G4CXApp (this=0x6c31e0, runMgr=0x6665d0) at /home/blyth/opticks/g4cx/tests/G4CXApp.h:158
    #8  0x000000000040a889 in G4CXApp::Create () at /home/blyth/opticks/g4cx/tests/G4CXApp.h:337
    #9  0x000000000040ab11 in G4CXApp::Main () at /home/blyth/opticks/g4cx/tests/G4CXApp.h:350
    #10 0x000000000040acaf in main (argc=1, argv=0x7fffffff48c8) at /home/blyth/opticks/g4cx/tests/G4CXTest.cc:13
    (gdb) 


FIXED : Issue 2 : reemission not happening on B side causing huge A-B chi2 : scint has to be before absorption
------------------------------------------------------------------------------------------------------------------

HMM : did I switch it off ? 


::

    ~/o/G4CXTest_GEOM.sh chi2
    ...

                      BASH_SOURCE : /data/blyth/junotop/opticks/g4cx/tests/../../sysrap/tests/sseq_index_test.sh 
                              SDIR : /data/blyth/junotop/opticks/sysrap/tests 
                               TMP : /data/blyth/opticks 
                        EXECUTABLE : G4CXTest 
                           VERSION : 98 
                              BASE : /data/blyth/opticks/GEOM/J_2024aug27 
                              GEOM : J_2024aug27 
                            LOGDIR : /data/blyth/opticks/GEOM/J_2024aug27/G4CXTest/ALL98 
                             AFOLD : /data/blyth/opticks/GEOM/J_2024aug27/G4CXTest/ALL98/A000 
                             BFOLD : /data/blyth/opticks/GEOM/J_2024aug27/G4CXTest/ALL98/B000 
                              FOLD : /data/blyth/opticks/sseq_index_test 
    a_path $AFOLD/seq.npy /data/blyth/opticks/GEOM/J_2024aug27/G4CXTest/ALL98/A000/seq.npy a_seq (1000000, 2, 2, )
    b_path $BFOLD/seq.npy /data/blyth/opticks/GEOM/J_2024aug27/G4CXTest/ALL98/B000/seq.npy b_seq (1000000, 2, 2, )
    AB
    [sseq_index_ab::desc u.size 118903 opt BRIEF mode 6sseq_index_ab_chi2::desc sum 72267.9580 ndf 1122.0000 sum/ndf    64.4099 sseq_index_ab_chi2_ABSUM_MIN:40.0000
    :r:`TO AB                                                                                            :  126549 252223 : 41697.7873 : Y :       2      6 : DEVIANT  `
    :r:`TO SC AB                                                                                         :   51434 101905 : 16612.3546 : Y :       4     27 : DEVIANT  `
        TO BT BT BT BT BT BT SD                                                                          :   70475  70075 :     1.1384 : Y :      18      2 :   
        TO BT BT BT BT BT BT SA                                                                          :   57091  56852 :     0.5013 : Y :       5      5 :   
    :r:`TO SC SC AB                                                                                      :   19993  39350 :  6314.0294 : Y :     137      7 : DEVIANT  `
        TO SC BT BT BT BT BT BT SD                                                                       :   35876  36034 :     0.3472 : Y :      58      1 :   
        TO SC BT BT BT BT BT BT SA                                                                       :   29663  29958 :     1.4596 : Y :     124      3 :   
    :r:`TO BT BT SA                                                                                      :   19822  18739 :    30.4165 : Y :      71     72 : DEVIANT  `
        TO RE AB                                                                                         :   18319     -1 :     0.0000 : N :       9     -1 : BZERO C2EXC  
        TO SC SC BT BT BT BT BT BT SD                                                                    :   15451  15549 :     0.3098 : Y :      19     81 :   
    :r:`TO SC SC SC AB                                                                                   :    7544  14559 :  2226.4048 : Y :      90     44 : DEVIANT  `
        TO SC SC BT BT BT BT BT BT SA                                                                    :   12785  12944 :     0.9826 : Y :      24    175 :   
        TO BT BT AB                                                                                      :   10955  11138 :     1.5158 : Y :      72     66 :   




No, but while investigated the UseGivenVelocity kludge I recall changing process order::

    P[blyth@localhost u4]$ git lg U4Physics.cc 
    * 3420a9d4a - placing all use of InstrumentedG4OpBoundaryProcess behind WITH_INSTRUMENTED_DEBUG gets standard G4OpBoundaryProcess to be used, which has no velocity issue in Geant4 1121 (7 months ago) <Simon C Blyth>
    * 7faff0a78 - add notes/issues/G4CXTest_raindrop_shows_Geant4_Process_Reorder_doesnt_fix_velocity_after_reflection_in_Geant4_1120.rst (7 months ago) <Simon C Blyth>
    * b58712977 - find that changing opticalphoton PostStepDoIt process order to do boundary after scintillation/reemission avoids the velocity issue and hence need for the UseGivenVelocity kludge (7 months ago) <Simon C Blyth>
    * 93c45d76a - investigate velocity after reflection (TIR or otherwise), find that UseGivenVelocity keeps that working as well as refraction (8 months ago) <Simon C Blyth>
    * 3e8c9b580 - implement sysrap/SPMTAccessor.h to provide C4IPMTAccessor interface on top of SPMT.h to allow U4Physics.cc to use real JUNO PMT info in standalone PMT testing such as with G4CXApp.h without depending on junosw or even the j/PMTSim extracts (12 months ago) <Simon C Blyth>
    * 00b7448c3 - fix BoundaryFlag zero assert WITH_CUSTOM4 NOT:WITH_PMTSIM using u4/U4PMTAccessor.h from u4/U4Physics.cc (1 year ago) <Simon C Blyth>
    * f92f8a597 - investigating U4Recorder/U4StepPoint BoundaryFlag zeros for WITH_CUSTOM4 NOT:WITH_PMTSIM and fix U4 SLOG logging by switching OPTICKS_U4 from PRIVATE to PUBLIC (1 year ago) <Simon C Blyth>
    * ef9479eb0 - start g4cx/tests/G4CXAppTest.sh based on u4/tests/U4SimulateTest.sh aiming for standalone bi-simulation, shrink MOCK_CURAND coverage and add without MOCK_CURAND methods to SGenerate.h to allow use of SGenerate.h when using CUDA (1 year, 3 months ago) <Simon C Blyth>



Issue 3 : deviant very simple histories : TO BT BT SA : at 1000/1M level 
--------------------------------------------------------------------------------


* :doc:`G4CXTest_GEOM_deviant_simple_history_TO_BT_BT_SA_at_1000_per_1M_level`


