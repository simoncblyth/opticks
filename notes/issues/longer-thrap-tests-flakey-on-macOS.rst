longer-thrap-tests-flakey-on-macOS
====================================

Two flakey ones are longer running ones::

    TCURANDTest
    TRngBufTest


Curiously after the GPU is "warmed up" by running with 
smaller loads they succeed.



::

    epsilon:thrustrap blyth$ om-test
    === om-test-one : thrustrap       /Users/blyth/opticks/thrustrap                               /usr/local/opticks/build/thrustrap                           
    Mon Apr  6 12:27:12 BST 2020
    Test project /usr/local/opticks/build/thrustrap
          Start  1: ThrustRapTest.TCURANDTest
     1/17 Test  #1: ThrustRapTest.TCURANDTest ......................   Passed    5.91 sec
          Start  2: ThrustRapTest.CBufSpecTest
     2/17 Test  #2: ThrustRapTest.CBufSpecTest .....................   Passed    0.94 sec
          Start  3: ThrustRapTest.TBufTest
     3/17 Test  #3: ThrustRapTest.TBufTest .........................   Passed    0.67 sec
          Start  4: ThrustRapTest.TRngBufTest
     4/17 Test  #4: ThrustRapTest.TRngBufTest ......................   Passed    1.65 sec
          Start  5: ThrustRapTest.expandTest
     5/17 Test  #5: ThrustRapTest.expandTest .......................   Passed    0.96 sec
          Start  6: ThrustRapTest.iexpandTest
     6/17 Test  #6: ThrustRapTest.iexpandTest ......................   Passed    0.68 sec
          Start  7: ThrustRapTest.issue628Test
     7/17 Test  #7: ThrustRapTest.issue628Test .....................   Passed    0.66 sec
          Start  8: ThrustRapTest.printfTest
     8/17 Test  #8: ThrustRapTest.printfTest .......................   Passed    0.60 sec
          Start  9: ThrustRapTest.repeated_rangeTest
     9/17 Test  #9: ThrustRapTest.repeated_rangeTest ...............   Passed    0.63 sec
          Start 10: ThrustRapTest.strided_rangeTest
    10/17 Test #10: ThrustRapTest.strided_rangeTest ................   Passed    0.66 sec
          Start 11: ThrustRapTest.strided_repeated_rangeTest
    11/17 Test #11: ThrustRapTest.strided_repeated_rangeTest .......   Passed    0.71 sec
          Start 12: ThrustRapTest.float2intTest
    12/17 Test #12: ThrustRapTest.float2intTest ....................   Passed    0.73 sec
          Start 13: ThrustRapTest.thrust_curand_estimate_pi
    13/17 Test #13: ThrustRapTest.thrust_curand_estimate_pi ........   Passed    0.84 sec
          Start 14: ThrustRapTest.thrust_curand_printf
    14/17 Test #14: ThrustRapTest.thrust_curand_printf .............   Passed    0.83 sec
          Start 15: ThrustRapTest.thrust_curand_printf_redirect
    15/17 Test #15: ThrustRapTest.thrust_curand_printf_redirect ....   Passed    0.78 sec
          Start 16: ThrustRapTest.thrust_curand_printf_redirect2
    16/17 Test #16: ThrustRapTest.thrust_curand_printf_redirect2 ...   Passed    0.75 sec
          Start 17: ThrustRapTest.TBuf4x4Test
    17/17 Test #17: ThrustRapTest.TBuf4x4Test ......................***Exception: Child aborted  0.97 sec
    2020-04-06 12:27:30.315 INFO  [18452725] [main@306] /usr/local/opticks/build/thrustrap/tests/TBuf4x4Test
    2020-04-06 12:27:30.316 INFO  [18452725] [test_copy4x4_ptr@210] (

