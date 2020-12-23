g4_1062_withfix-opticks-t-6-of-438-fails-charles-account
=========================================================

::

    FAILS:  6   / 438   :  Wed Dec 23 12:53:55 2020   
      42 /50  Test #42 : SysRapTest.SPPMTest                           ***Exception: SegFault         0.15   
      14 /112 Test #14 : NPYTest.ImageNPYTest                          Child aborted***Exception:     0.08   
      15 /112 Test #15 : NPYTest.ImageNPYConcatTest                    Child aborted***Exception:     0.27   

      Permissions errors from bare /tmp/paths.txt. Fix with SPath::Resolve using SPath::UserTmpDir

      22 /32  Test #22 : OptiXRapTest.interpolationTest                ***Failed                      6.32   

      MISSED PYTHONPATH=$HOME for "import opticks"

      21 /24  Test #21 : ExtG4Test.X4GDMLReadStructure2Test            ***Exception: SegFault         0.12   

      Did not handle lack of a gdmlpath argument

      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      0.27   
    epsilon:~ charles$ 



interpolationTest
------------------

Fails to run some python::

    epsilon:optixrap charles$ /opt/local/bin/python /Users/charles/local/opticks/bin/interpolationTest_interpol.py
    Traceback (most recent call last):
      File "/Users/charles/local/opticks/bin/interpolationTest_interpol.py", line 23, in <module>
        from opticks.ana.proplib import PropLib
    ImportError: No module named opticks.ana.proplib
    epsilon:optixrap charles$ 

Add line to example.opticks_config::

     43 export PYTHONPATH=$HOME   # needed to allow python scripts to "import opticks"


IntegrationTests.tboolean.box
---------------------------------

Works now, must have been PYTHONPATH too::

    epsilon:opticks charles$ cd integration
    epsilon:integration charles$ om-test
    === om-test-one : integration     /Users/charles/opticks/integration                           /Users/charles/local/opticks/build/integration               
    Wed Dec 23 15:40:15 GMT 2020
    Test project /Users/charles/local/opticks/build/integration
        Start 1: IntegrationTests.IntegrationTest
    1/2 Test #1: IntegrationTests.IntegrationTest ...   Passed    0.03 sec
        Start 2: IntegrationTests.tboolean.box
    2/2 Test #2: IntegrationTests.tboolean.box ......   Passed   16.49 sec

    100% tests passed, 0 tests failed out of 2

    Total Test time (real) =  16.53 sec
    Wed Dec 23 15:40:32 GMT 2020
    epsilon:integration charles$ 



