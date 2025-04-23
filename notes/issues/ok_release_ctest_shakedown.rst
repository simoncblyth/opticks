ok_release_ctest_shakedown
=============================


Issue 1 : ctest dont work from readonly dir : WORKAROUND ADDED FOR NEXT RELEASE
-----------------------------------------------------------------------------------

::

    A[blyth@localhost ~]$ opticks-install-tests-script-
    #!/bin/bash

    cd $(dirname $(realpath $BASH_SOURCE))

    # As ctest doesnt currently work from a readonly dir
    # this copies the released ctest metadata folders to TTMP
    # directory and runs the ctest from there
    #
    # ctest -N
    # ctest --output-on-failure

    ttmp=/tmp/$USER/opticks/tests
    TTMP=${TTMP:-$ttmp}
    mkdir -p $TTMP
    cp -r . ${TTMP}/

    cd $TTMP
    pwd

    ctest -N
    ctest --output-on-failure



Issue 2 : workstation gives one test fail : stale test ? FIXED FOR NEXT RELEASE
-------------------------------------------------------------------------------------

::

    A[blyth@localhost tests]$ ctest --output-on-failure --rerun-failed
    Test project /tmp/blyth/opticks/tests
        Start 29: SysRapTest.SCurandStateTest
    1/1 Test #29: SysRapTest.SCurandStateTest ......***Failed    0.01 sec
    2025-04-23 11:22:38.569 INFO  [2255607] [main@11] 
    /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.3.8/lib/SCurandStateTest: symbol lookup error: /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.3.8/lib/SCurandStateTest: undefined symbol: _ZN12SCurandStateC1EPKc


    0% tests passed, 1 tests failed out of 1

    Total Test time (real) =   0.01 sec

    The following tests FAILED:
         29 - SysRapTest.SCurandStateTest (Failed)
    Errors while running CTest
    :0A[blyth@localhost tests]$ pwd
    /tmp/blyth/opticks/tests
    A[blyth@localhost tests]$ 


    A[blyth@localhost opticks]$ find . -name SCurandStateTest.cc
    A[blyth@localhost opticks]$ 


    A[blyth@localhost opticks_Debug]$ find . -name SCurandStateTest
    ./el9_amd64_gcc11/Opticks-v0.3.6/lib/SCurandStateTest
    ./el9_amd64_gcc11/Opticks-v0.3.4/lib/SCurandStateTest
    ./el9_amd64_gcc11/Opticks-v0.3.8/lib/SCurandStateTest
    ./el9_amd64_gcc11/Opticks-v0.3.5/lib/SCurandStateTest
    ./Opticks-v0.2.7/x86_64--gcc11-geant4_10_04_p02-dbg/lib/SCurandStateTest
    ./lib/SCurandStateTest
    ./Opticks-v0.3.2/x86_64--gcc11-geant4_10_04_p02-dbg/lib/SCurandStateTest
    ./Opticks-v0.3.3/x86_64--gcc11-geant4_10_04_p02-dbg/lib/SCurandStateTest
    ./Opticks-v0.3.1/x86_64--gcc11-geant4_10_04_p02-dbg/lib/SCurandStateTest
    A[blyth@localhost opticks_Debug]$



Three stale binaries in lib::

      36 -rwxr-xr-x.  1 blyth blyth   35304 Apr 22 16:34 SArrTest
      76 -rwxr-xr-x.  1 blyth blyth   76120 Apr 22 16:34 SArTest
     148 -rwxr-xr-x.  1 blyth blyth  147608 Apr 22 16:34 S_get_option_Test
     204 -rwxr-xr-x.  1 blyth blyth  204960 Apr 22 16:34 SOKConfTest
     204 -rwxr-xr-x.  1 blyth blyth  207656 Apr 22 16:34 SSys2Test
      24 -rwxr-xr-x.  1 blyth blyth   21624 Apr 22 16:34 CPPVersionInteger
      24 -rwxr-xr-x.  1 blyth blyth   22880 Apr 22 16:34 Geant4VersionInteger
      36 -rwxr-xr-x.  1 blyth blyth   36624 Apr 22 16:34 OKConfTest
      24 -rwxr-xr-x.  1 blyth blyth   21896 Apr 22 16:34 OpticksVersionNumberTest

    1256 -rwxr-xr-x.  1 blyth blyth 1282664 Dec 18 11:13 QSim_Lifecycle_Test
    1112 -rwxr-xr-x.  1 blyth blyth 1137920 Dec 18 11:12 SRngSpecTest
     252 -rwxr-xr-x.  1 blyth blyth  256744 Nov  6 14:57 SCurandStateTest
 
   A[blyth@localhost opticks_Debug]$ 



::

    A[blyth@localhost ~]$ t opticks-prefix-find-stale
    opticks-prefix-find-stale () 
    { 
        local iwd=$PWD;
        cd $OPTICKS_PREFIX;
        pwd;
        local tops="lib lib64";
        local ref=lib/OKConfTest;
        for top in $tops;
        do
            echo $FUNCNAME find files in $top 10 minutes or more older than ref $ref;
            find $top -type f ! -newermt "$(date -r $ref) - 10 min";
        done;
        cd $iwd
    }

    A[blyth@localhost ~]$ opticks-prefix-find-stale
    /data1/blyth/local/opticks_Debug
    opticks-prefix-find-stale find files in lib 10 minutes or more older than ref lib/OKConfTest
    lib/SCurandStateTest
    lib/QSim_Lifecycle_Test
    lib/SRngSpecTest
    opticks-prefix-find-stale find files in lib64 10 minutes or more older than ref lib/OKConfTest
    A[blyth@localhost ~]$ 






Why is opticks-install-tests including a stale test ? Its not the problem is did not run opticks-install-extras
-----------------------------------------------------------------------------------------------------------------


/data1/blyth/local/opticks_Debug/tests/sysrap/tests/CTestTestfile.cmake::

     53 add_test(SysRapTest.SCFTest "SCFTest")
     54 set_tests_properties(SysRapTest.SCFTest PROPERTIES  _BACKTRACE_TRIPLES "/home/blyth/opticks/sysrap/tests/CMakeLists.txt;202;add_test;/home/blyth/opticks/sysrap/tests/CMakeLists.txt;0;")
     55 add_test(SysRapTest.SCurandStateTest "SCurandStateTest")
     56 set_tests_properties(SysRapTest.SCurandStateTest PROPERTIES  _BACKTRACE_TRIPLES "/home/blyth/opticks/sysrap/tests/CMakeLists.txt;202;add_test;/home/blyth/opticks/sysrap/tests/CMakeLists.txt;0;")
     57 add_test(SysRapTest.PLogTest "PLogTest")


::

    A[blyth@localhost tests]$ opticks-install-tests
                FUNCNAME : opticks-install-tests 
                    bdir : /data1/blyth/local/opticks_Debug/build 
                    dest : /data1/blyth/local/opticks_Debug/tests 
                  script : /data1/blyth/local/opticks_Debug/bin/CTestTestfile.py 
                    fold : /home/blyth 
    [2025-04-23 14:19:43,086] p2321283 {/home/blyth/opticks/bin/CMakeLists.py:198} INFO - home /home/blyth/opticks 
    [2025-04-23 14:19:43,105] p2321283 {/data1/blyth/local/opticks_Debug/bin/CTestTestfile.py:68} INFO - root /data1/blyth/local/opticks_Debug/build 
    [2025-04-23 14:19:43,105] p2321283 {/data1/blyth/local/opticks_Debug/bin/CTestTestfile.py:69} INFO - projs ['okconf', 'sysrap', 'CSG', 'qudarap', 'gdxml', 'u4', 'CSGOptiX', 'g4cx'] 
    [2025-04-23 14:19:43,110] p2321283 {/data1/blyth/local/opticks_Debug/bin/CTestTestfile.py:139} INFO - Copying CTestTestfile.cmake files from buildtree /data1/blyth/local/opticks_Debug/build into a new destination tree /data1/blyth/local/opticks_Debug/tests 
    [2025-04-23 14:19:43,110] p2321283 {/data1/blyth/local/opticks_Debug/bin/CTestTestfile.py:140} INFO - write testfile to /data1/blyth/local/opticks_Debug/tests/CTestTestfile.cmake 
    A[blyth@localhost tests]$ 


HUH: its not on rerunning the stale doesnt appear::

     53 add_test(SysRapTest.SCFTest "SCFTest")
     54 set_tests_properties(SysRapTest.SCFTest PROPERTIES  _BACKTRACE_TRIPLES "/home/blyth/opticks/sysrap/tests/CMakeLists.txt;215;add_test;/home/blyth/opticks/sysrap/tests/CMakeLists.txt;0;")
     55 add_test(SysRapTest.PLogTest "PLogTest")
     56 set_tests_properties(SysRapTest.PLogTest PROPERTIES  _BACKTRACE_TRIPLES "/home/blyth/opticks/sysrap/tests/CMakeLists.txt;215;add_test;/home/blyth/opticks/sysrap/tests/CMakeLists.txt;0;")
     57 add_test(SysRapTest.SLOG_Test "SLOG_Test")



::

    A[blyth@localhost issues]$ cd /data1/blyth/local/opticks_Debug
    A[blyth@localhost opticks_Debug]$ rm lib/SCurandStateTest lib/QSim_Lifecycle_Test lib/SRngSpecTest
    A[blyth@localhost opticks_Debug]$ 
    A[blyth@localhost opticks_Debug]$ 
    A[blyth@localhost opticks_Debug]$ opticks-prefix-find-stale
    /data1/blyth/local/opticks_Debug
    opticks-prefix-find-stale find files in lib 10 minutes or more older than ref lib/OKConfTest
    opticks-prefix-find-stale find files in lib64 10 minutes or more older than ref lib/OKConfTest
    A[blyth@localhost opticks_Debug]$ 



::

    252 okdist-install-extras()
    253 {
    254    local msg="=== $FUNCNAME :"
    255    local iwd=$PWD
    256 
    257    opticks-
    258    opticks-cd  ## install directory
    259 
    260    opticks-install-extras   ## avoid stale ctest by updating before release
    261    
    262    echo $msg write metadata
    263    okdist-install-metadata
    264    
    265    cd $iwd
    266 }  





Issue 3 : cluster running : one stale + geometry + g4 fails
----------------------------------------------------------------

Workstation running they all pass. On cluster 59 fails.::

    1289 72% tests passed, 59 tests failed out of 211
    1290 
    1291 Total Test time (real) = 419.61 sec
    1292 
    1293 The following tests FAILED:
    1294      29 - SysRapTest.SCurandStateTest (Failed)  ## expected : the stale test in this release

    1295     110 - SysRapTest.SSimTest (Failed)   ## looks like ancient geometry configured
    1296     111 - SysRapTest.SBndTest (Failed)

    1297     112 - CSGTest.CSGNodeTest (Failed)
    1298     116 - CSGTest.CSGPrimSpecTest (Failed)
    1299     117 - CSGTest.CSGPrimTest (Failed)
    1300     119 - CSGTest.CSGFoundryTest (Failed)
    1301     121 - CSGTest.CSGFoundry_getCenterExtent_Test (Failed)
    1302     122 - CSGTest.CSGFoundry_findSolidIdx_Test (Failed)
    1303     123 - CSGTest.CSGFoundry_CreateFromSimTest (Failed)
    1304     125 - CSGTest.CSGNameTest (Failed)
    1305     126 - CSGTest.CSGTargetTest (Failed)
    1306     127 - CSGTest.CSGTargetGlobalTest (Failed)
    1307     128 - CSGTest.CSGFoundry_MakeCenterExtentGensteps_Test (Failed)
    1308     129 - CSGTest.CSGFoundry_getFrame_Test (Failed)
    1309     130 - CSGTest.CSGFoundry_getFrameE_Test (Failed)
    1310     131 - CSGTest.CSGFoundry_getMeshName_Test (Failed)
    1311     134 - CSGTest.CSGFoundryLoadTest (Failed)
    1312     135 - CSGTest.CSGScanTest (Failed)
    1313     140 - CSGTest.CSGSimtraceTest (Failed)



SSimTest and SBndTest are both BASH_RUN_TEST_SOURCES using SSim::Load 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* WIP : changed to standard CFBaseFromGEOM geometry resolution for SSim::Load too

Examine the succeeding test on workstation::

    A[blyth@localhost tests]$ which SSimTest
    /data1/blyth/local/opticks_Debug/lib/SSimTest
    A[blyth@localhost tests]$ which STestRunner.sh
    /data1/blyth/local/opticks_Debug/bin/STestRunner.sh
    A[blyth@localhost tests]$ STestRunner.sh SSimTest > /tmp/out
    A[blyth@localhost tests]$ rc
    RC 0

::

   01                 HOME : /home/blyth
    2                  PWD : /home/blyth/opticks/sysrap/tests
    3                 GEOM : J_2025_mar24
    4          BASH_SOURCE : /data1/blyth/local/opticks_Debug/bin/STestRunner.sh
    5           EXECUTABLE : SSimTest
    6                 ARGS :
    7 [NPFold::desc
    8 NPFold::desc_subfold
    9  tot_items 12044
   10  folds 3882
   11  paths 3882
   12   0 [/]  stamp:0
   13   1 [//stree]  stamp:0
   14   2 [//stree/material]  stamp:0
   15   3 [//stree/material/Air]  stamp:0
   16   4 [//stree/material/Rock]  stamp:0
   17   5 [//stree/material/Galactic]  stamp:0
   18   6 [//stree/material/Steel]  stamp:0


::

    A[blyth@localhost tests]$ CSGFoundry=INFO STestRunner.sh SSimTest > /tmp/out2




STestRunner.sh HMM this is probably assuming default $HOME/.opticks/GEOM/$GEOM location to get geometry::

     54 EXECUTABLE="$1"
     55 shift
     56 ARGS="$@"
     57 
     58 
     59 
     60 geomscript=$HOME/.opticks/GEOM/GEOM.sh
     61 [ -s $geomscript ] && source $geomscript
     62 
     63 
     64 vars="HOME PWD GEOM BASH_SOURCE EXECUTABLE ARGS"
     65 for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done
     66 
     67 #env 
     68 $EXECUTABLE $@
     69 [ $? -ne 0 ] && echo $BASH_SOURCE : FAIL from $EXECUTABLE && exit 1




Yes::

     84 
     85 /**
     86 SSim::Load from persisted geometry  : used for testing 
     87 -------------------------------------------------------
     88  
     89 **/
     90 
     91 const char* SSim::DEFAULT = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry" ;
     92 
     93 SSim* SSim::Load(){ return Load_(DEFAULT) ; }
     94 
     95 SSim* SSim::Load_(const char* base_)
     96 {
     97     LOG(LEVEL) << "[" ;
     98     const char* base = spath::Resolve(base_ ? base_ : DEFAULT );
     99     LOG(LEVEL)
    100        << " base_ [" << ( base_ ? base_ : "-" ) << "]"
    101        << " base [" << ( base ? base : "-" ) << "]"
    102        ;
    103 
    104     SSim* sim = new SSim ;
    105     sim->load(base);    // reldir defaults to "SSim"
    106 
    107     LOG(LEVEL) << "]" ;
    108     return sim ;
    109 }



Could do::

     91 const char* SSim::DEFAULT = "${OPTICKS_DOTFOLD:-$HOME}/.opticks/GEOM/$GEOM/CSGFoundry" ;

But thats another envvar when already using::

     73     ## external geometry hookup  ## needs to be separate from other Opticks setup, can be combined in OJ  
     74     export GEOM=J25_3_0_Opticks_v0_3_5
     75     export ${GEOM}_CFBaseFromGEOM=/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.3.0_Opticks-v0.3.5/el9_amd64_gcc11/2025_04_14/.opticks/GEOM/$GEOM


But that is from old functionality SOpticksResource::CFBaseFromGEOM




Cluster ctest release run down to 11/211 FAIL
-------------------------------------------------

After setup geometry provision in /hpcfs/juno/junogpu/blyth/.opticks/GEOM/GEOM.sh + G4 env get to 11/211 FAIL::

     974 95% tests passed, 11 tests failed out of 211
     975 
     976 Total Test time (real) =  99.05 sec
     977 
     978 The following tests FAILED:
     979      29 - SysRapTest.SCurandStateTest (Failed)   ## expected

     980     110 - SysRapTest.SSimTest (Failed)

     981     139 - CSGTest.CSGQueryTest (Failed)

     982     160 - QUDARapTest.QSimTest (Failed)
     983     162 - QUDARapTest.QOpticalTest (Failed)
     984     166 - QUDARapTest.QSim_Lifecycle_Test (Failed)
     985     167 - QUDARapTest.QSimWithEventTest (Failed)

     986     184 - U4Test.U4GDMLReadTest (Failed)   
                           ## changed to U4GDML.h 
         15     //static constexpr const char* DefaultGDMLPath = "$UserGEOMDir/origin.gdml" ;  
         16     static constexpr const char* DefaultGDMLPath = "$CFBaseFromGEOM/origin.gdml" ;

     987     195 - U4Test.U4TraverseTest (Failed)
     988     210 - G4CXTest.G4CXRenderTest (Failed)
     989     211 - G4CXTest.G4CXOpticks_setGeometry_Test (Failed)
                           ## these four look like fail to find GDML

     990 okjob-tail : rc 8
     991 Wed Apr 23 07:49:26 PM CST 2025




Changes for release ctest have broken standard ctest
--------------------------------------------------------


Initially::

    FAILS:  5   / 217   :  Wed Apr 23 21:12:01 2025   
      107/108 Test #107: SysRapTest.SSimTest                           ***Failed                      0.09   
      28 /43  Test #28 : CSGTest.CSGQueryTest                          ***Failed                      0.10   
      18 /21  Test #18 : QUDARapTest.QOpticalTest                      ***Failed                      0.01   
      19 /21  Test #19 : QUDARapTest.QSimWithEventTest                 ***Failed                      0.38   
      20 /21  Test #20 : QUDARapTest.QSimTest                          ***Failed                      0.38   

Then::

    SLOW: tests taking longer that 15 seconds
      107/108 Test #107: SysRapTest.SSimTest                           Passed                         19.36  
      1  /2   Test #1  : G4CXTest.G4CXRenderTest                       Passed                         15.98  

    FAILS:  1   / 217   :  Wed Apr 23 21:37:09 2025   
      28 /43  Test #28 : CSGTest.CSGQueryTest                          ***Failed                      0.10  
      ## CSGMaker CSGFoundry GEOM  config needs tidy up    

