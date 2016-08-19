FIXED : Automated Interop Tests
================================

Former Issue
-------------

Testing of interop running currently requires
user interaction to dismiss OpenGL windows 
that pop up.

Workout a way to automatically do this, or 
even better avoid popping up a window and 
instead use some non-visible framebuffer.

This would then allow to implement tests that 
compare digests of interop and compute simulations.


How Fixed
------------

Bash functions *opticks-t* and *opticks-ti* set an 
option that ctest interprets and as a result sets 
envvar *CTEST_INTERACTIVE_DEBUG_MODE* in the 
interactive case.
This is detected in `SSys::IsCTestInteractiveDebugMode()` 
and honoured in *ggeoview/App.cc* by exiting 
prior to the render loop.

::

   --interactive-debug-mode 0   ## no interactivity 
   --interactive-debug-mode 1   ## interactive GUI windows OK


Actually already have *--noviz/-V** option
--------------------------------------------

::

   GGeoViewTest -V --save   


Detecting CTest Running ?
---------------------------

::

    simon:tests blyth$ sysrap-t -R SysRapTest.SEnvTest -V

    1: SSH_INFOFILE=/Users/blyth/.ssh-agent-info
    1: _=/opt/local/bin/ctest
    1: CTEST_INTERACTIVE_DEBUG_MODE=1


* https://cmake.org/Wiki/CTest_2.4.4_Docs

::

     --interactive-debug-mode [0|1]
       Set the interactive mode to 0 or 1.

       This option causes ctest to run tests in either an interactive mode or
       a non-interactive mode.  On Windows this means that in non-interactive
       mode, all system debug pop up windows are blocked.  In dashboard mode
       (Experimental, Nightly, Continuous), the default is non-interactive.
       When just running tests not for a dashboard the default is to allow
       popups and interactive debugging.


::

    sysrap-t --interactive-debug-mode 0 -R SysRapTest.SEnvTest -V  ## no envvar CTEST_INTERACTIVE_DEBUG_MODE in non-interactive mode

    1: DART_TEST_FROM_DART=1
    1: DASHBOARD_TEST_FROM_CTEST=3.4.1

    sysrap-t --interactive-debug-mode 1 -R SysRapTest.SEnvTest -V    ## this is the default

    1: _=/opt/local/bin/ctest
    1: CTEST_INTERACTIVE_DEBUG_MODE=1


Currently the windows popup whilst testing::

    simon:issues blyth$ ggeoview-t --interactive-debug-mode 0 
    Test project /usr/local/opticks/build/ggeoview
        Start 1: GGeoViewTest.flagsTest
    1/5 Test #1: GGeoViewTest.flagsTest ...........   Passed    0.02 sec
        Start 2: GGeoViewTest.OTracerTest
    2/5 Test #2: GGeoViewTest.OTracerTest .........   Passed    5.39 sec
        Start 3: GGeoViewTest.GGeoViewTest
    3/5 Test #3: GGeoViewTest.GGeoViewTest ........   Passed   15.31 sec
        Start 4: GGeoViewTest.LogTest
    4/5 Test #4: GGeoViewTest.LogTest .............   Passed    0.02 sec
        Start 5: GGeoViewTest.OpEngineTest
    5/5 Test #5: GGeoViewTest.OpEngineTest ........   Passed    1.09 sec

    100% tests passed, 0 tests failed out of 5

    Total Test time (real) =  21.83 sec
    opticks-t : use -V to show output
    simon:issues blyth$ 



