2017-10-19-axel-test-backtrace
===================================

::


    98% tests passed, 6 tests failed out of 254

    Total Test time (real) = 151.86 sec

    The following tests FAILED:
        134 - OpticksCoreTest.OpticksTest (SEGFAULT)          # PLOG issues
        243 - cfg4Test.CGDMLDetectorTest (OTHER_FAULT)
        244 - cfg4Test.CGeometryTest (OTHER_FAULT)
        250 - cfg4Test.CInterpolationTest (OTHER_FAULT)

        242 - cfg4Test.CTestDetectorTest (OTHER_FAULT)        # old style PmtInBox geometry needs overhaul
        226 - OptiXRapTest.OInterpolationTest (Failed)        # missing GBndLib.npy 

    Errors while running CTest
    opticks-t- : use -V to show output



PLOG test fails with gcc5
-----------------------------

* :doc:`PLOG-test-fails-with-gcc5`


FIXED : OInterpolationTest : analysis scripts need to use opticks_main otherwise depend on TMP envvar   
--------------------------------------------------------------------------------------------------------

::

    IOError: [Errno 2] No such file or directory: '$TMP/InterpolationTest/GBndLib/GBndLib.npy'
    2017-10-18 15:49:52.771 INFO  [3879] [SSys::run@46] python /home/gpu/opticks/optixrap/tests/OInterpolationTest_interpol.py rc_raw : 256 rc : 1
    2017-10-18 15:49:52.771 WARN  [3879] [SSys::run@52] SSys::run FAILED with  cmd python /home/gpu/opticks/optixrap/tests/OInterpolationTest_interpol.py possibly you need to set export PATH=$OPTICKS_HOME/ana:$OPTICKS_HOME/bin:/usr/local/opticks/lib:$PATH 
    [Thread 0x7fffeb874700 (LWP 3885) exited]
    [Thread 0x7fffec075700 (LWP 3884) exited]
    [Thread 0x7ffff7fb8780 (LWP 3879) exited]
    [Inferior 1 (process 3879) exited with code 01]
    (gdb) bt
    No stack.


Reproduced this with::

   unset TMP
   OInterpolationTest


If TMP is not defined then the analysis python scripts must use opticks_main,
wherein it gets defined internally within the python environ::

   from opticks.ana.base import opticks_main

::

    2017-10-19 14:27:01.598 INFO  [304911] [OContext::close@245] OContext::close m_cfg->apply() done.
    2017-10-19 14:27:04.979 INFO  [304911] [OContext::launch@322] OContext::launch LAUNCH time: 3.38112
    Traceback (most recent call last):
      File "/Users/blyth/opticks/optixrap/tests/OInterpolationTest_interpol.py", line 15, in <module>
        blib = PropLib.load_GBndLib(base)
      File "/Users/blyth/opticks/ana/proplib.py", line 96, in load_GBndLib
        t = np.load(os.path.expandvars(os.path.join(base,"GBndLib/GBndLib.npy")))
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/lib/npyio.py", line 369, in load
        fid = open(file, "rb")
    IOError: [Errno 2] No such file or directory: '$TMP/InterpolationTest/GBndLib/GBndLib.npy'
    2017-10-19 14:27:05.139 INFO  [304911] [SSys::run@46] python /Users/blyth/opticks/optixrap/tests/OInterpolationTest_interpol.py rc_raw : 256 rc : 1
    2017-10-19 14:27:05.139 WARN  [304911] [SSys::run@52] SSys::run FAILED with  cmd python /Users/blyth/opticks/optixrap/tests/OInterpolationTest_interpol.py possibly you need to set export PATH=$OPTICKS_HOME/ana:$OPTICKS_HOME/bin:/usr/local/opticks/lib:$PATH 
    simon:boostrap blyth$ 







gdb CTestDetectorTest : this one requires cfg4 overhaul
---------------------------------------------------------

::

    4294967295 bname Vacuum///Vacuum lv __dd__Geometry__RPC__lvRPCBarCham140xbf4c6a0
    CTestDetectorTest: /home/gpu/opticks/ggeo/GSurLib.cc:147: void GSurLib::examineSolidBndSurfaces(): Assertion `node == i' failed.

    Program received signal SIGABRT, Aborted.
    0x00007ffff5ca8428 in __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:54
    54	../sysdeps/unix/sysv/linux/raise.c: No such file or directory.
    (gdb) bt
    #0  0x00007ffff5ca8428 in __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:54
    #1  0x00007ffff5caa02a in __GI_abort () at abort.c:89
    #2  0x00007ffff5ca0bd7 in __assert_fail_base (fmt=<optimized out>, assertion=assertion@entry=0x7ffff7049d2a "node == i", file=file@entry=0x7ffff7049d08 "/home/gpu/opticks/ggeo/GSurLib.cc", 
        line=line@entry=147, function=function@entry=0x7ffff7049dc0 <GSurLib::examineSolidBndSurfaces()::__PRETTY_FUNCTION__> "void GSurLib::examineSolidBndSurfaces()") at assert.c:92
    #3  0x00007ffff5ca0c82 in __GI___assert_fail (assertion=0x7ffff7049d2a "node == i", file=0x7ffff7049d08 "/home/gpu/opticks/ggeo/GSurLib.cc", line=147, 
        function=0x7ffff7049dc0 <GSurLib::examineSolidBndSurfaces()::__PRETTY_FUNCTION__> "void GSurLib::examineSolidBndSurfaces()") at assert.c:101
    #4  0x00007ffff6fd83ac in GSurLib::examineSolidBndSurfaces (this=0x3024c00) at /home/gpu/opticks/ggeo/GSurLib.cc:147
    #5  0x00007ffff6fd7f18 in GSurLib::close (this=0x3024c00) at /home/gpu/opticks/ggeo/GSurLib.cc:93
    #6  0x00007ffff66aa3d6 in CDetector::attachSurfaces (this=0x30245e0) at /home/gpu/opticks/cfg4/CDetector.cc:244
    #7  0x00007ffff665fc8a in CGeometry::init (this=0x3023bd0) at /home/gpu/opticks/cfg4/CGeometry.cc:73
    #8  0x00007ffff665f9e4 in CGeometry::CGeometry (this=0x3023bd0, hub=0x7fffffffd2d0) at /home/gpu/opticks/cfg4/CGeometry.cc:39
    #9  0x00007ffff66c27ca in CG4::CG4 (this=0x7fffffffd350, hub=0x7fffffffd2d0) at /home/gpu/opticks/cfg4/CG4.cc:123
    #10 0x0000000000403e2f in main (argc=1, argv=0x7fffffffd998) at /home/gpu/opticks/cfg4/tests/CTestDetectorTest.cc:50




