opticks_ctest_shakedown_with_cuda131_on_rtx6000pro_server.rst
=================================================================


How to run ctest for an OK release on the server ?
-----------------------------------------------------


::

    L[blyth@junogpu001 oj]$ pwd
    /hpcfs/juno/junogpu/blyth/oj
    L[blyth@junogpu001 oj]$ source /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4/bashrc 
    === opticks-setup-geant4- : WARNING no OPTICKS_GEANT4_PREFIX : Geant4 will need be setup by other means
    L[blyth@junogpu001 oj]$ which opticks-t
    /usr/bin/which: no opticks-t in (/usr/share/Modules/bin:/usr/lib64/ccache:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/cvmfs/common.ihep.ac.cn/software/cctools:/opt/puppetlabs/bin:/usr/local/cuda-13.1/bin:/cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4/bin:/cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4/lib)
    L[blyth@junogpu001 oj]$ which ctest
    /usr/bin/ctest
    L[blyth@junogpu001 oj]$ echo $OPTICKS_PREFIX
    /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4
    L[blyth@junogpu001 oj]$ 



    L[blyth@junogpu001 tests]$ $OPTICKS_PREFIX/bin/ctest.sh
             BASH_SOURCE : /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4/bin/ctest.sh
                     arg : info_copy_run_ana
                  defarg : info_copy_run_ana
          OPTICKS_PREFIX : /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4
                    tdir : /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4/tests
                     tmp : /tmp/blyth/opticks
                     TMP : /tmp/blyth/opticks
                    TTMP : /tmp/blyth/opticks/tests
                     log : ctest.log
    /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4/bin/ctest.sh - copy ctest from /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4/tests to TTMP /tmp/blyth/opticks/tests
    [ /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4/bin/ctest.sh - run ctest with tee logging to /tmp/blyth/opticks/tests/ctest.log
    /tmp/blyth/opticks/tests
    [ do_ctest
    Tue Jun  2 05:27:31 PM CST 2026
    Test project /tmp/blyth/opticks/tests
      Test   #1: OKConfTest.OKConfTest
      Test   #2: OKConfTest.OpticksVersionNumberTest
      Test   #3: OKConfTest.Geant4VersionInteger
      Test   #4: OKConfTest.CPPVersionInteger
      Test   #5: SysRapTest.PythonImportTest
      Test   #6: SysRapTest.SOKConfTest
      Test   #7: SysRapTest.SArTest
      Test   #8: SysRapTest.SArrTest
      Test   #9: SysRapTest.SArgsTest
      Test  #10: SysRapTest.STimesTest
      Test  #11: SysRapTest.SEnvTest
      Test  #12: SysRapTest.SSysTest
      Test  #13: SysRapTest.SSys2Test
      Test  #14: SysRapTest.SSys3Test
      Test  #15: SysRapTest.SStrTest
      Test  #16: SysRapTest.SPathTest
      Test  #17: SysRapTest.STrancheTest
      Test  #18: SysRapTest.SVecTest
      ...



Log goes to /tmp/blyth/opticks/tests/ctest.log, that may dissapear::

    L[blyth@junogpu001 ~]$ cp /tmp/blyth/opticks/tests/ctest.log server-ctest.log

Geometry related fail::

            Start 149: CSGTest.CSGSimtraceRerunTest
    149/217 Test #149: CSGTest.CSGSimtraceRerunTest .............................   Passed    2.83 sec
            Start 150: CSGTest.CSGSimtraceSampleTest
    150/217 Test #150: CSGTest.CSGSimtraceSampleTest ............................   Passed    2.83 sec
            Start 151: CSGTest.CSGCopyTest
    151/217 Test #151: CSGTest.CSGCopyTest ......................................***Failed    5.39 sec
    /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4/bin/CSGTestRunner.sh - using external config for GEOM J25_3_0_Opticks_v0_3_5 J25_3_0_Opticks_v0_3_5_CFBaseFromGEOM
                    HOME : /hpcfs/juno/junogpu/blyth
                     PWD : /scratch/tmp/blyth/opticks/tests/CSG/tests
                    GEOM : J25_3_0_Opticks_v0_3_5
             BASH_SOURCE : /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4/bin/CSGTestRunner.sh
              EXECUTABLE : CSGCopyTest
                    ARGS : 
    2026-06-02 17:30:04.131 INFO  [223112] [main@16]  mode [K]
    2026-06-02 17:30:04.259 INFO  [223112] [SEventConfig::SetDevice@1904] SEventConfig::DescDevice
    name                             : NVIDIA RTX PRO 6000 Blackwell Server Edition
    totalGlobalMem_bytes             : 101975851008
    totalGlobalMem_GB                : 94
    HeuristicMaxSlot(VRAM)           : 776291136
    HeuristicMaxSlot(VRAM)/M         : 776
    HeuristicMaxSlot_Rounded(VRAM)   : 776000000
    MaxSlot/M                        : 180
    ModeLite                         : 0
    ModeMerge                        : 0

    2026-06-02 17:30:04.259 INFO  [223112] [SEventConfig::SetDevice@1919]  Configured_MaxSlot/M 180 Final_MaxSlot/M 180 HeuristicMaxSlot_Rounded/M 776 changed NO  DeviceName NVIDIA RTX PRO 6000 Blackwell Server Edition HasDevice YES
    (export OPTICKS_MAX_SLOT=0 # to use VRAM based HeuristicMaxPhoton) 
    stree::ImportArray array is null, label[prim_nidx.npy]
    stree::ImportArray array is null, label[nidx_prim.npy]
    stree::ImportNames array is null, label[prname.txt]
    (typically this means the stree.h serialization code has changed compared to the version used for saving the tree)
    stree::ImportNames array is null, label[soname.txt]
    (typically this means the stree.h serialization code has changed compared to the version used for saving the tree)
    2026-06-02 17:30:06.862 INFO  [223112] [main@28] env->desc()
     ELV         t 308 : 11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
    src->descELV(elv)
    CSGFoundry::descELV elv.num_bits 308 num_include 308 num_exclude 0 is_all_set 1
    INCLUDE:308

    p:  0:midx:  0:mn:sTopRock_domeAir
    p:  1:midx:  1:mn:sTopRock_dome
    p:  2:midx:  2:mn:sDomeRockBox
    p:  3:midx:  3:mn:PoolCoversub
    p:  4:midx:  4:mn:Upper_LS_tube
    p:  5:midx:  5:mn:Upper_Steel_tube



    2026-06-02 17:30:06.913 INFO  [223112] [CSGFoundry::CompareVec@704] prim sizeof(T) 64 data_match FAIL 
    2026-06-02 17:30:06.914 INFO  [223112] [CSGFoundry::CompareVec@708] prim sizeof(T) 64 byte_match FAIL 
    2026-06-02 17:30:06.914 FATAL [223112] [CSGFoundry::CompareVec@711]  mismatch FAIL for prim
     mismatch FAIL for prim a.size 3392 b.size 3392
    2026-06-02 17:30:06.920 FATAL [223112] [CSGFoundry::Compare@615]  mismatch FAIL 
    2026-06-02 17:30:06.920 INFO  [223112] [main@46]  src 0x11143920 dst 0x11150310 cf 2
    2026-06-02 17:30:06.920 FATAL [223112] [main@54]  UNEXPECTED DIFFERENCE  DEBUG WITH :
     ~/opticks/CSG/tests/CSGCopyTest.sh ana 
    CSGCopyTest: /home/blyth/opticks/CSG/tests/CSGCopyTest.cc:61: int main(int, char**): Assertion `cf == 0' failed.
    /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4/bin/CSGTestRunner.sh: line 58: 223112 Aborted                 (core dumped) $EXECUTABLE $@
    /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4/bin/CSGTestRunner.sh : FAIL from CSGCopyTest

            Start 152: CSGTest.CSGIntersectComparisonTest
    152/217 Test #152: CSGTest.CSGIntersectComparisonTest .......................   Passed    0.05 sec
            Start 153: CSGTest.distance_leaf_slab_test
    153/217 Test #153: CSGTest.distance_leaf_slab_test ..........................   Passed    0.03 sec
            Start 154: CSGTest.CSGSignedDistanceFieldTest


              GEOM : J25_3_0_Opticks_v0_3_5
         BASH_SOURCE : /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4/bin/U4TestRunner.sh
          EXECUTABLE : G4ThreeVectorTest
                ARGS : 


    G4ThreeVectorTest: error while loading shared libraries: libG4Tree.so: cannot open shared object file: No such file or directory
    /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4/bin/U4TestRunner.sh : FAIL from G4ThreeVectorTest

            Start 211: U4Test.U4PhysicsTableTest
    211/217 Test #211: U4Test.U4PhysicsTableTest ................................***Failed    0.04 sec
                    HOME : /hpcfs/juno/junogpu/blyth
                     PWD : /scratch/tmp/blyth/opticks/tests/u4/tests
                    GEOM : J25_3_0_Opticks_v0_3_5
             BASH_SOURCE : /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4/bin/U4TestRunner.sh
              EXECUTABLE : U4PhysicsTableTest
                    ARGS : 
    U4PhysicsTableTest: error while loading shared libraries: libG4Tree.so: cannot open shared object file: No such file or directory
    /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4/bin/U4TestRunner.sh : FAIL from U4PhysicsTableTest

            Start 212: CSGOptiXTest.CSGOptiXVersion
    212/217 Test #212: CSGOptiXTest.CSGOptiXVersion .............................   Passed    0.07 sec
            Start 213: CSGOptiXTest.CSGOptiXVersionTest
    213/217 Test #213: CSGOptiXTest.CSGOptiXVersionTest .........................   Passed    0.01 sec
            Start 214: CSGOptiXTest.CSGOptiXRenderTest
    214/217 Test #214: CSGOptiXTest.CSGOptiXRenderTest ..........................***Failed   13.44 sec
    /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4/bin/CXTestRunner.sh : NOT-FOUND A_CFBaseFromGEOM /tmp/blyth/opticks/G4CXOpticks_setGeometry_Test/J25_3_0_Opticks_v0_3_5 containing CSGFoundry/prim.npy
    /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4/bin/CXTestRunner.sh : NOT-FOUND B_CFBaseFromGEOM /hpcfs/juno/junogpu/blyth/.opticks/GEOM/J25_3_0_Opticks_v0_3_5 containing CSGFoundry/prim.npy
                    HOME : /hpcfs/juno/junogpu/blyth
                     PWD : /scratch/tmp/blyth/opticks/tests/CSGOptiX/tests
                    GEOM : J25_3_0_Opticks_v0_3_5
             BASH_SOURCE : /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4/bin/CXTestRunner.sh
              EXECUTABLE : CSGOptiXRenderTest
                    ARGS : 
    2026-06-02 17:30:22.317 INFO  [223370] [SEventConfig::SetDevice@1904] SEventConfig::DescDevice









::

    83% tests passed, 37 tests failed out of 217

    Total Test time (real) = 184.04 sec

    The following tests FAILED:
        151 - CSGTest.CSGCopyTest (Failed)
        176 - QUDARapTest.QSimWithEventTest (Failed)
        179 - QUDARapTest.QPMTTest (Failed)
        181 - U4Test.Deprecated_U4PhotonInfoTest (Failed)
        182 - U4Test.U4TrackInfoTest (Failed)
        183 - U4Test.U4TrackTest (Failed)
        184 - U4Test.U4Custom4Test (Failed)
        185 - U4Test.U4NistManagerTest (Failed)
        186 - U4Test.U4MaterialTest (Failed)
        187 - U4Test.U4MaterialPropertyVectorTest (Failed)
        188 - U4Test.U4GDMLTest (Failed)
        189 - U4Test.U4GDMLReadTest (Failed)
        190 - U4Test.U4PhysicalConstantsTest (Failed)
        191 - U4Test.U4RandomTest (Failed)
        192 - U4Test.U4UniformRandTest (Failed)
        193 - U4Test.U4EngineTest (Failed)
        194 - U4Test.U4RandomMonitorTest (Failed)
        195 - U4Test.U4RandomArrayTest (Failed)
        196 - U4Test.U4VolumeMakerTest (Failed)
        197 - U4Test.U4LogTest (Failed)
        198 - U4Test.U4RotationMatrixTest (Failed)
        199 - U4Test.U4TransformTest (Failed)
        200 - U4Test.U4TraverseTest (Failed)
        201 - U4Test.U4Material_MakePropertyFold_MakeTest (Failed)
        202 - U4Test.U4Material_MakePropertyFold_LoadTest (Failed)
        203 - U4Test.U4TouchableTest (Failed)
        204 - U4Test.U4SurfaceTest (Failed)
        205 - U4Test.U4SolidTest (Failed)
        206 - U4Test.U4SolidMakerTest (Failed)
        207 - U4Test.U4SensitiveDetectorTest (Failed)
        208 - U4Test.U4Debug_Test (Failed)
        209 - U4Test.U4Hit_Debug_Test (Failed)
        210 - U4Test.G4ThreeVectorTest (Failed)
        211 - U4Test.U4PhysicsTableTest (Failed)
        214 - CSGOptiXTest.CSGOptiXRenderTest (Failed)
        216 - G4CXTest.G4CXRenderTest (Failed)
        217 - G4CXTest.G4CXOpticks_setGeometry_Test (Failed)
    Errors while running CTest
    Tue Jun  2 05:30:35 PM CST 2026





Is this just failing to find geometry ?


Tis using some ancient GEOM.sh::

    L[blyth@junogpu001 ~]$ cat $HOME/.opticks/GEOM/GEOM.sh
    #!/bin/bash -l 
    notes(){ cat << EON
    ~/.opticks/GEOM/GEOM.sh
    =========================

    * THIS IS NOT UNDER SOURCE CONTROL BECAUSE IT IDENTIFIES A SPECIFIC GEOM 
    * NB keeping this as minimal as possible 
    * JUST EXPORT THE DEFAULT GEOM : NOTHING MORE 
    * any associated utility functions such as scp/grab/vi 
      should be kept under source control in the opticks.bash GEOM bash function 

    EON
    }

    #geom=FewPMT

    #geom=V1J009  # --no-guide-tube --debug-disable-xj
    #geom=V1J010   # --no-guide-tube --debug-disable-xj --debug-disable-sj --debug-disable-fa
    #geom=V1J011
    geom=J25_3_0_Opticks_v0_3_5

    #geom=RaindropRockAirWater

    export GEOM=$geom

    case $GEOM in
    J25_3_0_Opticks_v0_3_5) DOTFOLD=/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.3.0_Opticks-v0.3.5/el9_amd64_gcc11/2025_04_14 ;;
                         *) DOTFOLD=$HOME ;;
    esac

    export ${GEOM}_CFBaseFromGEOM=$DOTFOLD/.opticks/GEOM/$GEOM
    export       ${GEOM}_GDMLPath=$DOTFOLD/.opticks/GEOM/$GEOM/origin.gdml


    L[blyth@junogpu001 ~]$ 




After setup of ~/.opticks/GEOM/GEOM.sh using recent geometry from cvmfs : 33/217 FAILs
--------------------------------------------------------------------------------------------

::

    SLOW: tests taking longer that 5.0 seconds
      106/217 Test #106: SysRapTest.SGLFW_SOPTIX_Scene_test                      Passed                         61.96  
      113/217 Test #113: SysRapTest.SBndTest                                     Passed                         5.69   
      119/217 Test #119: CSGTest.CSGPrimTest                                     Passed                         14.31  
      133/217 Test #133: CSGTest.CSGFoundry_findSolidIdx_Test                    Passed                         5.33   
      136/217 Test #136: CSGTest.CSGFoundry_CreateFromSimTest                    Passed                         10.87  
      150/217 Test #150: CSGTest.CSGSimtraceSampleTest                           Passed                         5.34   


    SLOW: tests taking longer that 15.0 seconds
      106/217 Test #106: SysRapTest.SGLFW_SOPTIX_Scene_test                      Passed                         61.96  


    FAILS:  33  / 217   :  Tue Jun  2 18:28:30 2026  :  GEOM no-geom  
      181/217 Test #181: U4Test.Deprecated_U4PhotonInfoTest                      ***Failed                      0.00   
      182/217 Test #182: U4Test.U4TrackInfoTest                                  ***Failed                      0.00   
      183/217 Test #183: U4Test.U4TrackTest                                      ***Failed                      0.00   
      184/217 Test #184: U4Test.U4Custom4Test                                    ***Failed                      0.00   
      185/217 Test #185: U4Test.U4NistManagerTest                                ***Failed                      0.00   
      186/217 Test #186: U4Test.U4MaterialTest                                   ***Failed                      0.00   
      187/217 Test #187: U4Test.U4MaterialPropertyVectorTest                     ***Failed                      0.00   
      188/217 Test #188: U4Test.U4GDMLTest                                       ***Failed                      0.00   
      189/217 Test #189: U4Test.U4GDMLReadTest                                   ***Failed                      0.00   
      190/217 Test #190: U4Test.U4PhysicalConstantsTest                          ***Failed                      0.00   
      191/217 Test #191: U4Test.U4RandomTest                                     ***Failed                      0.00   
      192/217 Test #192: U4Test.U4UniformRandTest                                ***Failed                      0.00   
      193/217 Test #193: U4Test.U4EngineTest                                     ***Failed                      0.00   
      194/217 Test #194: U4Test.U4RandomMonitorTest                              ***Failed                      0.00   
      195/217 Test #195: U4Test.U4RandomArrayTest                                ***Failed                      0.00   
      196/217 Test #196: U4Test.U4VolumeMakerTest                                ***Failed                      0.00   
      197/217 Test #197: U4Test.U4LogTest                                        ***Failed                      0.00   
      198/217 Test #198: U4Test.U4RotationMatrixTest                             ***Failed                      0.00   
      199/217 Test #199: U4Test.U4TransformTest                                  ***Failed                      0.00   
      200/217 Test #200: U4Test.U4TraverseTest                                   ***Failed                      0.00   
      201/217 Test #201: U4Test.U4Material_MakePropertyFold_MakeTest             ***Failed                      0.00   
      202/217 Test #202: U4Test.U4Material_MakePropertyFold_LoadTest             ***Failed                      0.00   
      203/217 Test #203: U4Test.U4TouchableTest                                  ***Failed                      0.00   
      204/217 Test #204: U4Test.U4SurfaceTest                                    ***Failed                      0.00   
      205/217 Test #205: U4Test.U4SolidTest                                      ***Failed                      0.00   
      206/217 Test #206: U4Test.U4SolidMakerTest                                 ***Failed                      0.00   
      207/217 Test #207: U4Test.U4SensitiveDetectorTest                          ***Failed                      0.00   
      208/217 Test #208: U4Test.U4Debug_Test                                     ***Failed                      0.00   
      209/217 Test #209: U4Test.U4Hit_Debug_Test                                 ***Failed                      0.00   
      210/217 Test #210: U4Test.G4ThreeVectorTest                                ***Failed                      0.00   
      211/217 Test #211: U4Test.U4PhysicsTableTest                               ***Failed                      0.00   
      216/217 Test #216: G4CXTest.G4CXRenderTest                                 ***Failed                      0.00   
      217/217 Test #217: G4CXTest.G4CXOpticks_setGeometry_Test                   ***Failed                      0.00   



    ] /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.6.4/bin/ctest.sh - CTestLog.py /tmp/blyth/opticks/tests/ctest.log
    L[blyth@junogpu001 ~]$ 





After setup XercesC+CLHEP+Geant4+Custom4 env all ctest pass
--------------------------------------------------------------

::

   source /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.1.1/ExternalLibs/Xercesc/3.2.4/bashrc
   source /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.1.1/ExternalLibs/CLHEP/2.4.7.1/bashrc
   source /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.1.1/ExternalLibs/Geant4/10.04.p02.juno/bashrc
   source /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J26.1.1/ExternalLibs/custom4/0.1.8/bashrc
   $OPTICKS_PREFIX/bin/ctest.sh


Six tests are SLOW as for now are reading geometry from cvmfs::

    SLOW: tests taking longer that 5.0 seconds
      106/217 Test #106: SysRapTest.SGLFW_SOPTIX_Scene_test                      Passed                         6.48   
      119/217 Test #119: CSGTest.CSGPrimTest                                     Passed                         14.20  
      136/217 Test #136: CSGTest.CSGFoundry_CreateFromSimTest                    Passed                         10.86  
      137/217 Test #137: CSGTest.CSGFoundry_MakeCenterExtentGensteps_Test        Passed                         6.50   
      214/217 Test #214: CSGOptiXTest.CSGOptiXRenderTest                         Passed                         6.77   
      216/217 Test #216: G4CXTest.G4CXRenderTest                                 Passed                         5.04   


    SLOW: tests taking longer that 15.0 seconds

    FAILS:  0   / 217   :  Tue Jun  2 18:42:31 2026  :  GEOM no-geom  




Next test is cxs scan : try to reproduce G3 M1,M10,M20,..,M100 numbers on workstation
---------------------------------------------------------------------------------------

The presentation plot PNG is the below, from s5_background_image.txt::

    AB_Substamp_ALL_Etime_vs_Photon_rtx_gen1_gen3.png
    GEOM/J_2024aug27/CSGOptiXSMTest/ALL1_sreport/figs/sreport_ab/mpcap/AB_Substamp_ALL_Etime_vs_Photon_rtx_gen1_gen3.png 1280px_720px


The events clearly created by "TEST=medium_scan ~/o/cxs_min.sh"::

    elif [ "$TEST" == "medium_scan" ]; then

       opticks_num_event=12
       opticks_num_genstep=1x12
       opticks_num_photon=M1,1,10,20,30,40,50,60,70,80,90,100  # duplication of M1 is to workaround lack of metadata
       opticks_running_mode=SRM_TORCH
       #opticks_max_photon=M100

       # Remember multi-launch needs multiple gensteps in order to slice them up
       # such that each slice fits into VRAM.
       # So for big photon counts its vital to use multiple genstep.

But the python scripts controlled from cxs_min.sh only handle only one or two events.
So the scan result plotting must be using sreport machinery. 
That loads metadata from many events for convenient scan plotting.



CSGOptiX/tests/CSGOptiXSMTest.cc just does CSGOptiX::SimulateMain
--------------------------------------------------------------------

::

    int CSGOptiX::SimulateMain() // static
    {
        SProf::Add("CSGOptiX__SimulateMain_HEAD");
        SEventConfig::SetRGModeSimulate();
        CSGFoundry* fd = CSGFoundry::Load();
        CSGOptiX* cx = CSGOptiX::Create(fd) ;
        bool reset = true ;
        for(int i=0 ; i < SEventConfig::NumEvent() ; i++) cx->simulate(i, reset);
        SProf::UnsetTag();
        SProf::Add("CSGOptiX__SimulateMain_TAIL");
        SProf::Write();
        cx->write_Ctx_log();
        delete cx ; 
        return 0 ;
    }



Canonical script ~/o/cxs_min.sh : symbolic link to CSGOptiX/cxs_min.sh : is that the one ?
---------------------------------------------------------------------------------------------

Thats creating the events, sreport.sh OR sreport_ab.sh showing the scan ?



Find other use of executable CSGOptiXSMTest
--------------------------------------------------

::

    [lo] A[blyth@localhost tests]$ opticks-f CSGOptiXSMTest

    ./CSGOptiX/tests/CSGOptiXSMTest.cc:CSGOptiXSMTest : used from cxs_min.sh 
    ./CSGOptiX/tests/CMakeLists.txt:    CSGOptiXSMTest.cc
    ./CSGOptiX/cxt_precision.sh:        export RFOLD=/data1/blyth/tmp/GEOM/BigWaterPool/CSGOptiXSMTest/ALL98_Debug_Philox_${TEST}/A000
    ./CSGOptiX/CSGOptiX.h:    static int         SimulateMain();  // used by tests/CSGOptiXSMTest.cc
    ./CSGOptiX/CSGOptiX.cc:| cxs_min.sh  | tests/CSGOptiXSMTest.cc   | minimal simulate    |
    ./CSGOptiX/cxs_min.sh:This script runs the CSGOptiXSMTest executable which has no Geant4 dependency,
    ./CSGOptiX/cxs_min.sh:bin=CSGOptiXSMTest

    ./G4CXTest_GEOM.sh:#export BFOLD=$TMP/GEOM/$GEOM/CSGOptiXSMTest/ALL/A000  ## TMP OVERRIDE COMPARE A-WITH-A from CSGOptiXSMTest
    ./cxs_min.sh:This script runs the CSGOptiXSMTest executable which has no Geant4 dependency,
    ./cxs_min.sh:bin=CSGOptiXSMTest


    ./examples/UseGeometryShader/build.sh:  RECORD_FOLD=/data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL/p003 ~/o/examples/UseGeometryShader/build.sh
    ./examples/UseGeometryShader/build.sh:  RECORD_FOLD=/data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL/p003 ~/o/examples/UseGeometryShader/build.sh  ana
    ./examples/UseGeometryShader/build.sh: 1) record_fold=$TMP/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL/p001 ;;
    ./g4cx/tests/G4CXTest_GEOM.sh:#export BFOLD=$TMP/GEOM/$GEOM/CSGOptiXSMTest/ALL/A000  ## TMP OVERRIDE COMPARE A-WITH-A from CSGOptiXSMTest
    ./qudarap/tests/QEvtTest.sh:afold=/data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_merge_M10/A000
    ./qudarap/QEvt.cc:    #7  0x0000000000404a95 in main (argc=1, argv=0x7fffffffaf28) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXSMTest.cc:13

    ./sreport.sh:   /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL0
    ./sreport.sh:   /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL0_sreport
    ./sreport.sh:  N3) DIR=/data/blyth/opticks/GEOM/$GEOM/CSGOptiXSMTest/ALL2 ;;
    ./sreport.sh:  N6) DIR=/data/blyth/opticks/GEOM/$GEOM/CSGOptiXSMTest/ALL3 ;;
    ./sreport.sh:  N7) DIR=/data/blyth/opticks/GEOM/$GEOM/CSGOptiXSMTest/ALL1 ; LAB="TITAN RTX : Debug" ;; 
    ./sreport.sh:  N8) DIR=/data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1_Debug_Philox_large_evt ; LAB="TITAN RTX" ;; 
    ./sreport.sh:  N9) DIR=/data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1_Debug_Philox_vlarge_evt ; LAB="TITAN RTX" ;;
    ./sreport.sh:  A7) DIR=/data1/blyth/tmp/GEOM/$GEOM/CSGOptiXSMTest/ALL1    ; LAB="Ada RTX 5000 : Debug" ;;
    ./sreport.sh:  S7) DIR=/data/simon/opticks/GEOM/$GEOM/CSGOptiXSMTest/ALL1 ; LAB="TITAN RTX : Release" ;; 

    ./sysrap/NPU.hh:   dirpath : /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL0
    ./sysrap/NPU.hh:   returns : /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/sreport
    ./sysrap/NPU.hh:   dirpath : /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL0
    ./sysrap/NPU.hh:   returns : /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL0_sreport

    ./sysrap/tests/sreport_abcd.sh:      N7) echo /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1 ;; 
    ./sysrap/tests/sreport_abcd.sh:      A7) echo /data1/blyth/tmp/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1    ;;
    ./sysrap/tests/sreport_abcd.sh:      S7) echo /data/simon/opticks/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1 ;; 
    ./sysrap/tests/sreport_abcd.sh:      N8) echo /data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan ;; 
    ./sysrap/tests/sreport_abcd.sh:      A8) echo /data1/blyth/tmp/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan ;; 

    ./sysrap/tests/sseq_test.sh:executable=CSGOptiXSMTest

    ./sysrap/tests/sreport.sh:   /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL0
    ./sysrap/tests/sreport.sh:   /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL0_sreport
    ./sysrap/tests/sreport.sh:  N3) DIR=/data/blyth/opticks/GEOM/$GEOM/CSGOptiXSMTest/ALL2 ;;
    ./sysrap/tests/sreport.sh:  N6) DIR=/data/blyth/opticks/GEOM/$GEOM/CSGOptiXSMTest/ALL3 ;;
    ./sysrap/tests/sreport.sh:  N7) DIR=/data/blyth/opticks/GEOM/$GEOM/CSGOptiXSMTest/ALL1 ; LAB="TITAN RTX : Debug" ;; 
    ./sysrap/tests/sreport.sh:  N8) DIR=/data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1_Debug_Philox_large_evt ; LAB="TITAN RTX" ;; 
    ./sysrap/tests/sreport.sh:  N9) DIR=/data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1_Debug_Philox_vlarge_evt ; LAB="TITAN RTX" ;;
    ./sysrap/tests/sreport.sh:  A7) DIR=/data1/blyth/tmp/GEOM/$GEOM/CSGOptiXSMTest/ALL1    ; LAB="Ada RTX 5000 : Debug" ;;
    ./sysrap/tests/sreport.sh:  S7) DIR=/data/simon/opticks/GEOM/$GEOM/CSGOptiXSMTest/ALL1 ; LAB="TITAN RTX : Release" ;; 

    ./sysrap/tests/sreport.py:            ax.set_xlabel("cxr_min.sh (CSGOptiXSMTest) Process Run Time[s]", fontsize=20 ); 

    ./sysrap/tests/sreport_ab.sh:      N7) echo /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1 ;; 
    ./sysrap/tests/sreport_ab.sh:      A7) echo /data1/blyth/tmp/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1    ;;
    ./sysrap/tests/sreport_ab.sh:      S7) echo /data/simon/opticks/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1 ;; 
    ./sysrap/tests/sreport_ab.sh:      N8) echo /data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan ;; 
    ./sysrap/tests/sreport_ab.sh:      A8) echo /data1/blyth/tmp/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan ;; 

    ./sysrap/tests/SEvtTest.sh:export SEQPATH=/data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL4/A000/seq.npy
    ./sysrap/tests/SPM_test.sh:   export AFOLD=/data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_merge/A000
    ./sysrap/tests/sdigest_duplicate_test/sdigest_duplicate_test.sh:#hitfold=/data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvlarge_evt/A000
    ./sysrap/tests/sdigest_duplicate_test/sdigest_duplicate_test.sh:hitfold=/data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvvvlarge_evt/A000
    ./sysrap/tests/sdigest_test.sh:   export HITFOLD=/data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvlarge_evt/A000
    ./sysrap/tests/sseq_index_test.sh:    #executable=CSGOptiXSMTest

    ./sreport_ab.sh:      N7) echo /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1 ;; 
    ./sreport_ab.sh:      A7) echo /data1/blyth/tmp/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1    ;;
    ./sreport_ab.sh:      S7) echo /data/simon/opticks/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1 ;; 
    ./sreport_ab.sh:      N8) echo /data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan ;; 
    ./sreport_ab.sh:      A8) echo /data1/blyth/tmp/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan ;; 
    ./sreport_abcd.sh:      N7) echo /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1 ;; 
    ./sreport_abcd.sh:      A7) echo /data1/blyth/tmp/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1    ;;
    ./sreport_abcd.sh:      S7) echo /data/simon/opticks/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1 ;; 
    ./sreport_abcd.sh:      N8) echo /data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan ;; 
    ./sreport_abcd.sh:      A8) echo /data1/blyth/tmp/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan ;; 
    [lo] A[blyth@localhost opticks]$ 



BINGO : sreport_ab.sh IS THE ONE
------------------------------------


::

   ~/o/sreport_ab.sh

   A=N7 B=A7 ~/o/sreport_ab.sh

   A=N7 B=A7 PLOT=Substamp_ALL_Etime_vs_Photon ~/o/sreport_ab.sh

   A=N7 B=A7 PLOT=AB_Substamp_ALL_Etime_vs_Photon ~/o/sreport_ab.sh
       ## commandline that reproduces chep 2024 v0 fig 5 
       ## (XORWOW with 100M state loading)

   A=N8 B=A8 PLOT=AB_Substamp_ALL_Etime_vs_Photon ~/o/sreport_ab.sh
       ## commandline showing Philox scan
       ## (Philox with inline curand_init, no loading)


But those codes correspond to directories not present on new laptop::

    resolve(){
    case $1 in
      N7) echo /data/blyth/opticks/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1 ;;
      A7) echo    /data1/blyth/tmp/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1    ;;
      S7) echo /data/simon/opticks/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1 ;;

      N8) echo /data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan ;;
      A8) echo    /data1/blyth/tmp/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan ;;
    esac
   }


The /data dirs are from "PD" (TITAN RTX machine), but too big at 7G each to grab fully if not needed.
Looks like just need sibling dirs with _sreport appended : they are tiny at 200K each.

* Did this via additions to ~/o/sreport_ab.sh reproducing the analysis on new laptop


Now need to reproduce the cxs_min.sh stage, writing N9 A9
-----------------------------------------------------------



