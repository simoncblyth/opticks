opticks-t-2020-04-15-fails-3-of-420
=====================================


After fixes : 1-of-420
------------------------

opticks-tl::

    SLOW: tests taking longer that 15 seconds
      1  /1   Test #1  : OKG4Test.OKG4Test                             Passed                         20.99  


    FAILS:  1   / 420   :  Wed Apr 15 12:08:01 2020   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      12.43  
    epsilon:opticks blyth$ 


::

   cd integration
   om-test 
         ## shows the remaining problem is an "alignment" failure



3-of-420
----------

::

    SLOW: tests taking longer that 15 seconds
      1  /1   Test #1  : OKG4Test.OKG4Test                             Passed                         21.19  


    FAILS:  3   / 420   :  Wed Apr 15 10:30:28 2020   
      19 /26  Test #19 : OptiXRapTest.interpolationTest                ***Failed                      6.46   
      30 /34  Test #30 : CFG4Test.CAlignEngineTest                     ***Exception: Child aborted    0.11   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      0.55   
    epsilon:opticks blyth$ 



CAlignEngineTest : FIXED size consistency with TRngBufTest 
-------------------------------------------------------------


IntegrationTests.tboolean.box on issue FIXED but still ana FAIL
------------------------------------------------------------------

Initial fail to find module fixed, but still giving non-zero ana RC.


::

    cd integration
    om-cd
    ctest  

::

    epsilon:integration blyth$ LV=box tboolean.sh --generateoverride 10000
    ====== /Users/blyth/opticks/bin/tboolean.sh --generateoverride 10000 ====== PWD /Users/blyth/opticks/integration =================
    tboolean-lv --generateoverride 10000
    === tboolean-lv : tboolean-box
    Traceback (most recent call last):
      File "<stdin>", line 3, in <module>
    ImportError: No module named opticks.ana.main
    === tboolean-box : testconfig


Looks like the environment of the script misses stuff from invoking commandline ?
But can get it there by explicitly giving it::

    epsilon:integration blyth$ LV=box PYTHONPATH=$PYTHONPATH tboolean.sh --generateoverride 10000

Cause of this was an omitted "export" on the PYTHONPATH setting to HOME, so it worked 
when run directly but not via a script.  


::

    epsilon:tests blyth$ ctest -R box
    Test project /usr/local/opticks/build/integration/tests
        Start 2: IntegrationTests.tboolean.box
    1/1 Test #2: IntegrationTests.tboolean.box ....***Failed   14.09 sec



Check log to see the ana fail::

    epsilon:tests blyth$ tail -10  OKG4Test.log
    2020-04-15 12:09:44.529 NONE  [23967545] [OpticksViz::renderLoop@541] enter runloop 
    2020-04-15 12:09:44.533 NONE  [23967545] [OpticksViz::renderLoop@546] after frame.show() 
    2020-04-15 12:09:44.640 INFO  [23967545] [Animator::Summary@424] Composition::gui setup Animation T0:  OFF 0/0/    0.0000
    2020-04-15 12:09:44.640 INFO  [23967545] [Animator::Summary@424] Composition::initRotator T0:  OFF 0/0/    0.0000
    2020-04-15 12:09:46.118 INFO  [23967545] [Frame::key_pressed@817] Frame::key_pressed escape
    2020-04-15 12:09:46.118 NONE  [23967545] [OpticksViz::renderLoop@581]  renderlooplimit 0 count 135295
    2020-04-15 12:09:46.118 FATAL [23967545] [Opticks::dumpRC@227]  rc 7 rcmsg : OpticksAna::run non-zero RC from ana script
    2020-04-15 12:09:46.118 INFO  [23967545] [main@32]  RC 7
    2020-04-15 12:09:46.145 INFO  [23967545] [CG4::cleanup@473] [
    2020-04-15 12:09:46.146 INFO  [23967545] [CG4::cleanup@475] ]
    epsilon:tests blyth$ pwd
    /usr/local/opticks/build/integration/tests
    epsilon:tests blyth$ 



interpolationTest : FIXED apparently flakey issue of PYTHONPATH envvar, twas a missing export
--------------------------------------------------------------------------------------------------

::

    2020-04-15 11:02:59.619 INFO  [23876070] [interpolationTest::ana@178]  m_script interpolationTest_interpol.py path /usr/local/opticks/bin/interpolationTest_interpol.py
    Traceback (most recent call last):
      File "/usr/local/opticks/bin/interpolationTest_interpol.py", line 23, in <module>
        from opticks.ana.proplib import PropLib
    ImportError: No module named opticks.ana.proplib
    2020-04-15 11:02:59.757 INFO  [23876070] [SSys::run@91] python /usr/local/opticks/bin/interpolationTest_interpol.py rc_raw : 256 rc : 1
    2020-04-15 11:02:59.757 ERROR [23876070] [SSys::run@98] FAILED with  cmd python /usr/local/opticks/bin/interpolationTest_interpol.py RC 1
    2020-04-15 11:02:59.757 INFO  [23876070] [interpolationTest::ana@184]  RC 1
    epsilon:optixrap blyth$ python -c "from opticks.ana.proplib import PropLib"
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
    ImportError: No module named opticks.ana.proplib
    epsilon:optixrap blyth$ echo $PYTHONPATH
    /Users/blyth
    epsilon:optixrap blyth$ python -c "import opticks"
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
    ImportError: No module named opticks
    epsilon:optixrap blyth$ PYTHONPATH=$HOME python -c "import opticks"
    epsilon:optixrap blyth$ echo $PYTHONPATH
    /Users/blyth
    epsilon:optixrap blyth$ PYTHONPATH=$HOME python -c "from opticks.ana.proplib import PropLib"
    epsilon:optixrap blyth$ 

    epsilon:optixrap blyth$ python /usr/local/opticks/bin/interpolationTest_interpol.py
    Traceback (most recent call last):
      File "/usr/local/opticks/bin/interpolationTest_interpol.py", line 23, in <module>
        from opticks.ana.proplib import PropLib
    ImportError: No module named opticks.ana.proplib


    epsilon:optixrap blyth$ PYTHONPATH=$HOME python /usr/local/opticks/bin/interpolationTest_interpol.py
    [2020-04-15 11:06:55,706] p25698 {np_load             :nload.py  :95} WARNING  - np_load path_:$TMP/interpolationTest/GBndLib/GBndLib.npy path:/tmp/blyth/opticks/interpolationTest/GBndLib/GBndLib.npy DOES NOT EXIST 
    [2020-04-15 11:06:55,706] p25698 {np_load             :nload.py  :95} WARNING  - np_load path_:$TMP/interpolationTest/GBndLib/GBndLibOptical.npy path:/tmp/blyth/opticks/interpolationTest/GBndLib/GBndLibOptical.npy DOES NOT EXIST 
    [2020-04-15 11:06:55,706] p25698 {load_GBndLib        :proplib.py:119} WARNING  - missing GBndLib data : cannot create blib Proplib
    [2020-04-15 11:06:55,706] p25698 {<module>            :interpolationTest_interpol.py:39} WARNING  - failed to load blib GPropLib from base:$TMP/interpolationTest 
    epsilon:optixrap blyth$ 
    epsilon:optixrap blyth$ rc
    RC 0
    epsilon:optixrap blyth$ 


Huh something flakey about PYTHONPATH envvar being seen by the script ?


