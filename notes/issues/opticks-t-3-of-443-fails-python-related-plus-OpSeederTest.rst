opticks-t-3-of-443-fails-python-related-plus-OpSeederTest
==========================================================

::

    SLOW: tests taking longer that 15 seconds


    FAILS:  3   / 443   :  Mon Feb 15 03:28:00 2021   
      22 /32  Test #22 : OptiXRapTest.interpolationTest                ***Failed                      6.50   
      2  /5   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     10.17  
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      1.25   
    [blyth@localhost opticks]$ 


FIXED by controlling PYTHONPATH and adapting for new trivial kernel
------------------------------------------------------------------------

::

    SLOW: tests taking longer that 15 seconds
      2  /2   Test #2  : IntegrationTests.tboolean.box                 Passed                         16.23  


    FAILS:  0   / 443   :  Mon Feb 15 04:16:40 2021   




python/numpy issue
----------------------

::

    2021-02-15 03:25:28.372 INFO  [377813] [interpolationTest::launch@165]  save  base $TMP/optixrap/interpolationTest name interpolationTest_interpol.npy
    2021-02-15 03:25:28.452 INFO  [377813] [SSys::RunPythonScript@544]  script interpolationTest_interpol.py script_path /home/blyth/local/opticks/bin/interpolationTest_interpol.py python_executable /home/blyth/local/env/tools/conda/miniconda3/bin/python3
    /home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/numpy/__init__.py:138: UserWarning: mkl-service package failed to import, therefore Intel(R) MKL initialization ensuring its correct out-of-the box operation under condition when Gnu OpenMP had already been loaded by Python process is not assured. Please install mkl-service package, see http://github.com/IntelPython/mkl-service
    Traceback (most recent call last):
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/numpy/core/__init__.py", line 22, in <module>
        from . import multiarray
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/numpy/core/multiarray.py", line 9, in <module>
        import functools
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/functools.py", line 21, in <module>
        from collections import namedtuple
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/collections/__init__.py", line 21, in <module>
        from operator import itemgetter as _itemgetter, eq as _eq
    ImportError: dynamic module does not define module export function (PyInit_operator)

    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
      File "/home/blyth/local/opticks/bin/interpolationTest_interpol.py", line 22, in <module>
        import os,sys, numpy as np, logging
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/numpy/__init__.py", line 140, in <module>
        from . import core
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/numpy/core/__init__.py", line 48, in <module>
        raise ImportError(msg)
    ImportError: 

    IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

    Importing the numpy C-extensions failed. This error can happen for
    many reasons, often due to issues with your setup or how NumPy was
    installed.

    We have compiled some common reasons and troubleshooting tips at:

        https://numpy.org/devdocs/user/troubleshooting-importerror.html

    Please note and check the following:

      * The Python version is: Python3.7 from "/home/blyth/local/env/tools/conda/miniconda3/bin/python3"
      * The NumPy version is: "1.19.1"

    and make sure that they are the versions you expect.
    Please carefully study the documentation linked above for further help.

    Original error was: dynamic module does not define module export function (PyInit_operator)

    2021-02-15 03:25:28.524 INFO  [377813] [SSys::run@100] /home/blyth/local/env/tools/conda/miniconda3/bin/python3 /home/blyth/local/opticks/bin/interpolationTest_interpol.py  rc_raw : 256 rc : 1
    2021-02-15 03:25:28.525 ERROR [377813] [SSys::run@107] FAILED with  cmd /home/blyth/local/env/tools/conda/miniconda3/bin/python3 /home/blyth/local/opticks/bin/interpolationTest_interpol.py  RC 1
    2021-02-15 03:25:28.525 INFO  [377813] [SSys::RunPythonScript@551]  RC 1






::

    2/2 Test #2: IntegrationTests.tboolean.box ......***Failed    1.25 sec
    .bashrc OPTICKS_MODE dev TERM_ORIG xterm-256color TERM xterm-256color
    /home/blyth/junotop/ExternalLibs/Opticks/0.1.0/bashrc : no OPTICKS_TOP : OPTICKS_MODE dev
    ====== /home/blyth/local/opticks/bin/tboolean.sh --generateoverride 10000 ====== PWD /home/blyth/local/opticks/build/integration/tests =================
    tboolean-lv --generateoverride 10000
    === tboolean-lv : tboolean-box cmdline --generateoverride 10000
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/logging/__init__.py", line 26, in <module>
        import sys, os, time, io, traceback, warnings, weakref, collections.abc
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/traceback.py", line 3, in <module>
        import collections
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/collections/__init__.py", line 21, in <module>
        from operator import itemgetter as _itemgetter, eq as _eq
    ImportError: dynamic module does not define module export function (PyInit_operator)
    === tboolean-box : testconfig

    tboolean-info






OpSeederTest
---------------

::

    2/5 Test #2: OKOPTest.OpSeederTest ............Child aborted***Exception:  10.17 sec
    ...
     num_remainder_volumes 4486 num_instanced_volumes 7744 num_remainder_volumes + num_instanced_volumes 12230 num_total_faces 483996 num_total_faces_woi 2533452 (woi:without instancing) 
       0 pts Y  GPts.NumPt  4486 lvIdx ( 248 247 21 0 7 6 3 2 3 2 ... 237 238 239 240 241 242 243 244 245)
       1 pts Y  GPts.NumPt     1 lvIdx ( 1)
       2 pts Y  GPts.NumPt     1 lvIdx ( 197)
       3 pts Y  GPts.NumPt     1 lvIdx ( 195)
       4 pts Y  GPts.NumPt     1 lvIdx ( 198)
       5 pts Y  GPts.NumPt     5 lvIdx ( 47 46 43 44 45)

    2021-02-15 03:25:45.970 INFO  [379536] [OGeo::convert@294] [ nmm 6
    2021-02-15 03:25:46.148 INFO  [379536] [OGeo::convert@307] ] nmm 6
    2021-02-15 03:25:46.154 ERROR [379536] [cuRANDWrapper::setItems@154] CAUTION : are resizing the launch sequence 
    2021-02-15 03:25:47.059 FATAL [379536] [OpticksGen::targetGenstep@349] node_index from GenstepNPY is -1 (dummy frame), resetting to 0
    2021-02-15 03:25:47.065 INFO  [379536] [OpSeeder::seedPhotonsFromGenstepsViaOptiX@174] SEEDING TO SEED BUF  
    2021-02-15 03:25:47.785 INFO  [379536] [OPropagator::prelaunch@197] 0 : (0;0,0) 
    2021-02-15 03:25:47.785 INFO  [379536] [BTimes::dump@183] OPropagator::prelaunch
                  validate000                 0.008139
                   compile000              7.00001e-06
                 prelaunch000                 0.583014
    2021-02-15 03:25:47.785 INFO  [379536] [OPropagator::launch@268] LAUNCH NOW   printLaunchIndex ( -1 -1 -1) -
    2021-02-15 03:25:47.785 INFO  [379536] [OPropagator::launch@277] LAUNCH DONE
    2021-02-15 03:25:47.785 INFO  [379536] [OPropagator::launch@279] 0 : (0;100,1) 
    2021-02-15 03:25:47.785 INFO  [379536] [BTimes::dump@183] OPropagator::launch
                    launch001                 0.000255
    2021-02-15 03:25:47.786 INFO  [379536] [OEvent::downloadHits@471]  nhit 1 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    2021-02-15 03:25:47.786 INFO  [379536] [OEvent::downloadHiys@506]  nhiy 1 --dbghit N hitmask 0x40 SD SURFACE_DETECT
    2021-02-15 03:25:47.786 INFO  [379536] [TrivialCheckNPY::dump@117] OpSeederTest entryCode T photons 100,4,4 gensteps 10,6,4
    2021-02-15 03:25:47.786 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:0 label:(indices.u.w)genstep_offset i:0 u:64 uconstant*s:0
    2021-02-15 03:25:47.786 FATAL [379536] [TrivialCheckNPY::checkItemValue@234]  step 0[:,3,3]    (indices.u.w)genstep_offset FAIL 1
    2021-02-15 03:25:47.786 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:1 label:(indices.u.w)genstep_offset i:10 u:0 uconstant*s:6
    2021-02-15 03:25:47.786 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:1 label:(indices.u.w)genstep_offset i:11 u:0 uconstant*s:6
    2021-02-15 03:25:47.786 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:1 label:(indices.u.w)genstep_offset i:12 u:0 uconstant*s:6
    2021-02-15 03:25:47.786 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:1 label:(indices.u.w)genstep_offset i:13 u:0 uconstant*s:6
    2021-02-15 03:25:47.786 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:1 label:(indices.u.w)genstep_offset i:14 u:0 uconstant*s:6
    2021-02-15 03:25:47.786 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:1 label:(indices.u.w)genstep_offset i:15 u:0 uconstant*s:6
    2021-02-15 03:25:47.786 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:1 label:(indices.u.w)genstep_offset i:16 u:0 uconstant*s:6
    2021-02-15 03:25:47.786 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:1 label:(indices.u.w)genstep_offset i:17 u:0 uconstant*s:6
    2021-02-15 03:25:47.786 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:1 label:(indices.u.w)genstep_offset i:18 u:0 uconstant*s:6
    2021-02-15 03:25:47.786 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:1 label:(indices.u.w)genstep_offset i:19 u:0 uconstant*s:6
    ...
    2021-02-15 03:25:47.787 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:9 label:(indices.u.w)genstep_offset i:90 u:0 uconstant*s:54
    2021-02-15 03:25:47.787 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:9 label:(indices.u.w)genstep_offset i:91 u:0 uconstant*s:54
    2021-02-15 03:25:47.787 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:9 label:(indices.u.w)genstep_offset i:92 u:0 uconstant*s:54
    2021-02-15 03:25:47.787 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:9 label:(indices.u.w)genstep_offset i:93 u:0 uconstant*s:54
    2021-02-15 03:25:47.787 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:9 label:(indices.u.w)genstep_offset i:94 u:0 uconstant*s:54
    2021-02-15 03:25:47.787 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:9 label:(indices.u.w)genstep_offset i:95 u:0 uconstant*s:54
    2021-02-15 03:25:47.787 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:9 label:(indices.u.w)genstep_offset i:96 u:0 uconstant*s:54
    2021-02-15 03:25:47.787 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:9 label:(indices.u.w)genstep_offset i:97 u:0 uconstant*s:54
    2021-02-15 03:25:47.787 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:9 label:(indices.u.w)genstep_offset i:98 u:0 uconstant*s:54
    2021-02-15 03:25:47.787 WARN  [379536] [TrivialCheckNPY::checkItemValue@208] FAIL checkItemValue IS_UCONSTANT_SCALED  istep:9 label:(indices.u.w)genstep_offset i:99 u:0 uconstant*s:54
    2021-02-15 03:25:47.787 FATAL [379536] [TrivialCheckNPY::checkItemValue@234]  step 9[:,3,3]    (indices.u.w)genstep_offset FAIL 10
    2021-02-15 03:25:47.787 FATAL [379536] [OpSeederTest::OpSeederTest@117] seedDebugCheck FAIL 91
    OpSeederTest: /home/blyth/opticks/okop/tests/OpSeederTest.cc:118: OpSeederTest::OpSeederTest(int, char**): Assertion `sdc == 0' failed.


* this fail probably due to recent changes to the trivial kernel 


::

    2719 int OpticksEvent::seedDebugCheck(const char* msg)
    2720 {
    2721     // This can only be used with specific debug entry points 
    2722     // that write seeds as uint into the photon buffer
    2723     //
    2724     //     * entryCode T    TRIVIAL
    2725     //     * entryCode D    DUMPSEED
    2726 
    2727     assert(m_photon_data && m_photon_data->hasData());
    2728     assert(m_genstep_data && m_genstep_data->hasData());
    2729 
    2730     TrivialCheckNPY chk(m_photon_data, m_genstep_data, m_ok->getEntryCode());
    2731     return chk.check(msg);
    2732 }
    2733 





::

    [blyth@localhost issues]$ python3 -c "import numpy"
    /home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/numpy/__init__.py:138: UserWarning: mkl-service package failed to import, therefore Intel(R) MKL initialization ensuring its correct out-of-the box operation under condition when Gnu OpenMP had already been loaded by Python process is not assured. Please install mkl-service package, see http://github.com/IntelPython/mkl-service
    Traceback (most recent call last):
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/numpy/core/__init__.py", line 22, in <module>
        from . import multiarray
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/numpy/core/multiarray.py", line 9, in <module>
        import functools
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/functools.py", line 21, in <module>
        from collections import namedtuple
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/collections/__init__.py", line 21, in <module>
        from operator import itemgetter as _itemgetter, eq as _eq
    ImportError: dynamic module does not define module export function (PyInit_operator)

    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/numpy/__init__.py", line 140, in <module>
        from . import core
      File "/home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/numpy/core/__init__.py", line 48, in <module>
        raise ImportError(msg)
    ImportError: 

    IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

    Importing the numpy C-extensions failed. This error can happen for
    many reasons, often due to issues with your setup or how NumPy was
    installed.

    We have compiled some common reasons and troubleshooting tips at:

        https://numpy.org/devdocs/user/troubleshooting-importerror.html

    Please note and check the following:

      * The Python version is: Python3.7 from "/home/blyth/local/env/tools/conda/miniconda3/bin/python3"
      * The NumPy version is: "1.19.1"

    and make sure that they are the versions you expect.
    Please carefully study the documentation linked above for further help.

    Original error was: dynamic module does not define module export function (PyInit_operator)

    [blyth@localhost issues]$ 



* https://github.com/numpy/numpy/issues/15390


::

    [blyth@localhost issues]$ unset PYTHONPATH
    [blyth@localhost issues]$ python3 -c "import numpy"
    [blyth@localhost issues]$ 

    [blyth@localhost issues]$ python -c "import numpy"
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
    ImportError: No module named numpy
    [blyth@localhost issues]$ python3 -c "import numpy"
    [blyth@localhost issues]$ python3 -c "import opticks"
    [blyth@localhost issues]$ 



Fix this python issue by controlling the PYTHONPATH in ~/.opticks_config::

    094 
     95 
     96 opticks-
     97 opticks-setup > /dev/null  # source setup script which appends the Opticks and externals prefixes to CMAKE_PREFIX_PATH etc..
     98 [ $? -ne 0 ] && echo ERROR sourcing opticks-setup.sh at  $OPTICKS_PREFIX/bin/opticks-setup.sh && sleep 10000000
     99 
    100 # PYTHONPATH is needed to allow python scripts to "import opticks"
    101 # without this some of the opticks-t tests will fail
    102 export PYTHONPATH=$(dirname $OPTICKS_HOME)        
    103 
    104 



