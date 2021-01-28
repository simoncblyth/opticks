debugging_failing_tests
==========================


.. contents:: Table of Contents https://simoncblyth.bitbucket.io/opticks/docs/misc/debugging_failing_tests.html
   :depth: 3


*opticks-t* reporting of test fails
------------------------------------------


The *opticks-t* function ends by listing any failing tests, eg::

    FAILS:  4   / 440   :  Wed Jan 27 00:08:47 2021   
      9  /51  Test #9  : SysRapTest.SPathTest                          ***Exception: Other            0.25   
      22 /32  Test #22 : OptiXRapTest.interpolationTest                ***Failed                      10.87  
      32 /32  Test #32 : OptiXRapTest.intersectAnalyticTest.iaConvexpolyhedronTest ***Exception: Other            6.41   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      2.94   



Running Tests Individually
----------------------------

When tests fail it is helpful to run tests 
individually to see the output, for example.
Most Opticks tests should already be in your PATH and can be run directly with no arguments, for example::

    [blyth@localhost ~]$ which interpolationTest
    ~/local/opticks/lib/interpolationTest

    [blyth@localhost ~]$ which SPathTest 
    ~/local/opticks/lib/SPathTest

    [blyth@localhost ~]$ SPathTest 
    test_Stem@35: 
    test_GetHomePath@44: 
    ...
    [blyth@localhost ~]$ echo $? # return code from previous command, will be non-zero after running a failed test
    0
         

Running simple tests under gdb
-----------------------------------------------

Running failing tests in the *gdb* debugger enables the 
error site to be found precisely by collecting 
the backtrace.::

     gdb $(which interpolationTest)
     gdb > r   # "run"
     ...       # on reaching the error the debugger will stop   
     gdb > bt  # "backtrace" 

The backtrace provides the call stack of the place where the error occurs. 
This should be copy/pasted into emails to describe the location of the error. 


Running tests which require arguments under gdb
--------------------------------------------------

OptiXRapTest.intersectAnalyticTest.iaConvexpolyhedronTest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


A small number of the over 400 tests require arguments. These tests have triple dotted names such as::

    OptiXRapTest.intersectAnalyticTest.iaConvexpolyhedronTest
    IntegrationTests.tboolean.box 

To run these under gdb requires::

     [blyth@localhost ~]$ gdb --args $(which intersectAnalyticTest) --cu iaConvexpolyhedronTest.cu

To find out what the arguments should be look at the **CMakeLists.txt** from the tests directory 
of the subproject and also look for **.sh** scripts in the tests directory which may have 
example commandlines.






