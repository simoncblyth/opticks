With opticks-cmake-overhaul all thrap tests failing cudaGetDevice
==================================================================


All fail for same error
------------------------

::

    libc++abi.dylib: terminating with uncaught exception of type thrust::system::system_error: get_max_shared_memory_per_block :failed to cudaGetDevice: CUDA driver version is insufficient for CUDA runtime version


* https://github.com/kozyilmaz/nheqminer-macos/issues/16



Difference between the built and installed thrap binaries ?
--------------------------------------------------------------

* ctest : runs the build dir binaries 



Switching Between Opticks versions
-------------------------------------

Switch between opticks versions by changing OPTICKS_HOME in .bash_profile and starting new bash tab::

    319 export OPTICKS_HOME=$HOME/opticks
    320 #export OPTICKS_HOME=$HOME/opticks-cmake-overhaul
    321 
    322 opticks-(){  [ -r $OPTICKS_HOME/opticks.bash ] && . $OPTICKS_HOME/opticks.bash && opticks-env $* && opticks-export ; }
    323 


Old Opticks::

    epsilon:issues blyth$ which CBufSpecTest
    /usr/local/opticks/lib/CBufSpecTest

New Opticks::

    epsilon:issues blyth$ which CBufSpecTest
    /usr/local/opticks-cmake-overhaul/lib/CBufSpecTest

    * also possible there is difference between integrated and subproj CMake builds ?

    * somehow the integrated and proj builds somehow getting different nvcc flags ?






Huh building subproj with new Opticks doesnt have the issue ?
-----------------------------------------------------------------


::

    thrap-cd
    ./go.sh

    epsilon:thrustrap blyth$ thrap-t 
    Thu May 24 14:13:12 HKT 2018
    Test project /usr/local/opticks-cmake-overhaul/build/thrustrap
          Start  1: ThrustRapTest.CBufSpecTest
     1/15 Test  #1: ThrustRapTest.CBufSpecTest .....................   Passed    0.98 sec
          Start  2: ThrustRapTest.TBufTest
     2/15 Test  #2: ThrustRapTest.TBufTest .........................   Passed    1.00 sec
          Start  3: ThrustRapTest.TRngBufTest
     3/15 Test  #3: ThrustRapTest.TRngBufTest ......................   Passed    2.16 sec
          Start  4: ThrustRapTest.expandTest
     4/15 Test  #4: ThrustRapTest.expandTest .......................   Passed    1.04 sec
          Start  5: ThrustRapTest.iexpandTest
     5/15 Test  #5: ThrustRapTest.iexpandTest ......................   Passed    0.95 sec
          Start  6: ThrustRapTest.issue628Test
     6/15 Test  #6: ThrustRapTest.issue628Test .....................   Passed    0.92 sec
          Start  7: ThrustRapTest.printfTest
     7/15 Test  #7: ThrustRapTest.printfTest .......................   Passed    0.96 sec
          Start  8: ThrustRapTest.repeated_rangeTest
     8/15 Test  #8: ThrustRapTest.repeated_rangeTest ...............   Passed    1.17 sec
          Start  9: ThrustRapTest.strided_rangeTest
     9/15 Test  #9: ThrustRapTest.strided_rangeTest ................   Passed    0.98 sec
          Start 10: ThrustRapTest.strided_repeated_rangeTest
    10/15 Test #10: ThrustRapTest.strided_repeated_rangeTest .......   Passed    1.22 sec
          Start 11: ThrustRapTest.float2intTest
    11/15 Test #11: ThrustRapTest.float2intTest ....................   Passed    1.22 sec
          Start 12: ThrustRapTest.thrust_curand_estimate_pi
    12/15 Test #12: ThrustRapTest.thrust_curand_estimate_pi ........   Passed    1.32 sec
          Start 13: ThrustRapTest.thrust_curand_printf
    13/15 Test #13: ThrustRapTest.thrust_curand_printf .............   Passed    0.92 sec
          Start 14: ThrustRapTest.thrust_curand_printf_redirect
    14/15 Test #14: ThrustRapTest.thrust_curand_printf_redirect ....   Passed    1.13 sec
          Start 15: ThrustRapTest.thrust_curand_printf_redirect2
    15/15 Test #15: ThrustRapTest.thrust_curand_printf_redirect2 ...   Passed    0.95 sec

    100% tests passed, 0 tests failed out of 15

    Total Test time (real) =  16.91 sec
    Thu May 24 14:13:29 HKT 2018
    === opticks-t- : use -V to show output, ctest output written to /usr/local/opticks-cmake-overhaul/build/thrustrap/ctest.log
    epsilon:thrustrap blyth$ which CBufSpecTest
    /usr/local/opticks-cmake-overhaul/lib/CBufSpecTest
    epsilon:thrustrap blyth$ 



Old opticks
----------------

With the old Opticks::

    epsilon:opticks blyth$ thrap-t
    Thu May 24 13:59:22 HKT 2018
    Test project /usr/local/opticks/build/thrustrap
          Start  1: ThrustRapTest.CBufSpecTest
     1/15 Test  #1: ThrustRapTest.CBufSpecTest .....................   Passed    0.85 sec
          Start  2: ThrustRapTest.TBufTest
     2/15 Test  #2: ThrustRapTest.TBufTest .........................   Passed    0.93 sec
          Start  3: ThrustRapTest.TRngBufTest
     3/15 Test  #3: ThrustRapTest.TRngBufTest ......................   Passed    1.73 sec
          Start  4: ThrustRapTest.expandTest
     4/15 Test  #4: ThrustRapTest.expandTest .......................   Passed    0.87 sec
          Start  5: ThrustRapTest.iexpandTest
     5/15 Test  #5: ThrustRapTest.iexpandTest ......................   Passed    0.92 sec
          Start  6: ThrustRapTest.issue628Test
     6/15 Test  #6: ThrustRapTest.issue628Test .....................   Passed    0.94 sec
          Start  7: ThrustRapTest.printfTest
     7/15 Test  #7: ThrustRapTest.printfTest .......................   Passed    1.07 sec
          Start  8: ThrustRapTest.repeated_rangeTest
     8/15 Test  #8: ThrustRapTest.repeated_rangeTest ...............   Passed    1.00 sec
          Start  9: ThrustRapTest.strided_rangeTest
     9/15 Test  #9: ThrustRapTest.strided_rangeTest ................   Passed    0.95 sec
          Start 10: ThrustRapTest.strided_repeated_rangeTest
    10/15 Test #10: ThrustRapTest.strided_repeated_rangeTest .......   Passed    1.05 sec
          Start 11: ThrustRapTest.float2intTest
    11/15 Test #11: ThrustRapTest.float2intTest ....................   Passed    1.02 sec
          Start 12: ThrustRapTest.thrust_curand_estimate_pi
    12/15 Test #12: ThrustRapTest.thrust_curand_estimate_pi ........   Passed    1.37 sec
          Start 13: ThrustRapTest.thrust_curand_printf
    13/15 Test #13: ThrustRapTest.thrust_curand_printf .............   Passed    0.95 sec
          Start 14: ThrustRapTest.thrust_curand_printf_redirect
    14/15 Test #14: ThrustRapTest.thrust_curand_printf_redirect ....   Passed    1.13 sec
          Start 15: ThrustRapTest.thrust_curand_printf_redirect2
    15/15 Test #15: ThrustRapTest.thrust_curand_printf_redirect2 ...   Passed    1.16 sec

    100% tests passed, 0 tests failed out of 15

    Total Test time (real) =  15.96 sec
    Thu May 24 13:59:38 HKT 2018
    === opticks-t- : use -V to show output, ctest output written to /usr/local/opticks/build/thrustrap/ctest.log
    epsilon:opticks blyth$ 


New Opticks
-------------

::

    cd /tmp/build/thrap
    ctest

    0% tests passed, 15 tests failed out of 15

    Total Test time (real) =   0.42 sec

    The following tests FAILED:
          1 - ThrustRapTest.CBufSpecTest (Child aborted)
          2 - ThrustRapTest.TBufTest (Child aborted)
          3 - ThrustRapTest.TRngBufTest (Child aborted)
          4 - ThrustRapTest.expandTest (Child aborted)
          5 - ThrustRapTest.iexpandTest (Child aborted)
          6 - ThrustRapTest.issue628Test (Child aborted)
          7 - ThrustRapTest.printfTest (Child aborted)
          8 - ThrustRapTest.repeated_rangeTest (Child aborted)
          9 - ThrustRapTest.strided_rangeTest (Child aborted)
         10 - ThrustRapTest.strided_repeated_rangeTest (Child aborted)
         11 - ThrustRapTest.float2intTest (Child aborted)
         12 - ThrustRapTest.thrust_curand_estimate_pi (Child aborted)
         13 - ThrustRapTest.thrust_curand_printf (Child aborted)
         14 - ThrustRapTest.thrust_curand_printf_redirect (Child aborted)
         15 - ThrustRapTest.thrust_curand_printf_redirect2 (Child aborted)
    Errors while running CTest
    epsilon:thrustrap blyth$ CBufSpecTest
    libc++abi.dylib: terminating with uncaught exception of type thrust::system::system_error: get_max_shared_memory_per_block :failed to cudaGetDevice: CUDA driver version is insufficient for CUDA runtime version
    Abort trap: 6




