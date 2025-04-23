ok_release_ctest_shakedown
=============================


Issue 1 : ctest dont work from readonly dir
-----------------------------------------------

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



Issue 2 : one test fail : stale test ?
---------------------------------------------

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
    A[blyth@localhost tests]$ pwd
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




