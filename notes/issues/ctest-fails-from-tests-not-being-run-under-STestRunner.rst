FIXED : ctest-fails-from-tests-not-being-run-under-STestRunner.rst
======================================================================

Overview
-----------

Problem was the tests were not being run under the STestRunner.sh
so there was no geometry environment setup causing the SEGV.



Issue
-------

::

    FAILS:  2   / 219   :  Thu Jul  3 16:18:10 2025  :  GEOM J25_4_0_opticks_Debug  
      84 /110 Test #84 : SysRapTest.SSceneLoadTest                               ***Exception: SegFault         0.08   
      103/110 Test #103: SysRapTest.SGLFW_SOPTIX_Scene_test                      ***Exception: SegFault         0.09   



Two tests SEGV under ctest but not otherwise ?
-----------------------------------------------

::

    (ok) A[blyth@localhost tests]$ ctest -R SysRapTest.SSceneLoadTest --output-on-failure
    Test project /data1/blyth/local/opticks_Debug/build/sysrap/tests
        Start 84: SysRapTest.SSceneLoadTest
    1/1 Test #84: SysRapTest.SSceneLoadTest ........***Exception: SegFault  0.08 sec


    0% tests passed, 1 tests failed out of 1

    Total Test time (real) =   0.08 sec

    The following tests FAILED:
         84 - SysRapTest.SSceneLoadTest (SEGFAULT)
    Errors while running CTest
    (ok) A[blyth@localhost tests]$ 



    (ok) A[blyth@localhost tests]$  ctest -R SysRapTest.SGLFW_SOPTIX_Scene_test --output-on-failure
    Test project /data1/blyth/local/opticks_Debug/build/sysrap/tests
        Start 103: SysRapTest.SGLFW_SOPTIX_Scene_test
    1/1 Test #103: SysRapTest.SGLFW_SOPTIX_Scene_test ...***Exception: SegFault  0.10 sec


    0% tests passed, 1 tests failed out of 1

    Total Test time (real) =   0.10 sec

    The following tests FAILED:
        103 - SysRapTest.SGLFW_SOPTIX_Scene_test (SEGFAULT)
    Errors while running CTest
    (ok) A[blyth@localhost tests]$ 



Under runner does not fail
-----------------------------

::

    (ok) A[blyth@localhost tests]$ STestRunner.sh SSceneLoadTest
                    HOME : /home/blyth
                     PWD : /data1/blyth/local/opticks_Debug/build/sysrap/tests
                    GEOM : J25_4_0_opticks_Debug
             BASH_SOURCE : /data1/blyth/local/opticks_Debug/bin/STestRunner.sh
              EXECUTABLE : SSceneLoadTest
                    ARGS : 
    SceneLoadTest.main mismatch 0


    ok) A[blyth@localhost tests]$ STestRunner.sh "gdb SSceneLoadTest"
                    HOME : /home/blyth
                     PWD : /data1/blyth/local/opticks_Debug/build/sysrap/tests
                    GEOM : J25_4_0_opticks_Debug
             BASH_SOURCE : /data1/blyth/local/opticks_Debug/bin/STestRunner.sh
              EXECUTABLE : gdb SSceneLoadTest
                    ARGS : 
    GNU gdb (AlmaLinux) 14.2-3.el9
    ...
    (gdb) r
    Starting program: /data1/blyth/local/opticks_Debug/build/sysrap/tests/SSceneLoadTest 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    SceneLoadTest.main mismatch 0
    [Inferior 1 (process 736599) exited normally]
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-125.el9_5.3.alma.1.x86_64 libgcc-11.5.0-5.el9_5.alma.1.x86_64 libstdc++-11.5.0-5.el9_5.alma.1.x86_64 openssl-libs-3.2.2-6.el9_5.1.x86_64
    (gdb) 



