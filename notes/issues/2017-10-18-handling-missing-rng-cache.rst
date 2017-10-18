2017-10-18-handling-missing-rng-cache
=======================================

SG Fails : last 4 are due to missing RNG cache
--------------------------------------------------

::

    98% tests passed, 5 tests failed out of 240

    Total Test time (real) = 460.49 sec

    The following tests FAILED:
        226 - OptiXRapTest.OInterpolationTest (Failed)
        227 - OptiXRapTest.ORayleighTest (OTHER_FAULT)
        231 - OKOPTest.OpSeederTest (OTHER_FAULT)
        236 - OKTest.OKTest (OTHER_FAULT)
        239 - OKTest.VizTest (OTHER_FAULT)
    Errors while running CTest
    opticks-t- : use -V to show output

    [simon@localhost opticks]$ uname -a
    Linux localhost.localdomain 2.6.32-431.el6.x86_64 #1 SMP Thu Nov 21 13:35:52 CST 2013 x86_64 x86_64 x86_64 GNU/Linux
    [simon@localhost opticks]$ 


Why no RNG cache ?
---------------------

::

    [simon@localhost opticks]$ ll /usr/local/opticks/installcache/
    total 4
    drwxrwxr-x. 2 simon simon 4096 Oct 18 19:46 PTX
    [simon@localhost opticks]$ 
    [simon@localhost opticks]$ ll /usr/local/opticks/installcache/PTX/
    total 1908
    -rw-r--r--. 1 simon simon  30117 Sep 22 00:26 OptiXRap_generated_axisTest.cu.ptx
    -rw-r--r--. 1 simon simon  42997 Sep 22 00:26 OptiXRap_generated_boundaryLookupTest.cu.ptx


After running opticks-prepare-installcache
--------------------------------------------

::

    simon@localhost opticks]$ ll /usr/local/opticks/installcache/
    total 12
    drwxrwxr-x. 2 simon simon 4096 Oct 18 20:44 OKC
    drwxrwxr-x. 2 simon simon 4096 Oct 18 19:46 PTX
    drwxrwxr-x. 2 simon simon 4096 Oct 18 20:44 RNG
    [simon@localhost opticks]$ ll /usr/local/opticks/installcache/OKC/
    total 16
    -rw-rw-r--. 1 simon simon 260 Oct 18 20:44 GFlagIndexLocal.ini
    -rw-rw-r--. 1 simon simon 260 Oct 18 20:44 GFlagIndexSource.ini
    -rw-rw-r--. 1 simon simon 260 Oct 18 20:44 GFlagsLocal.ini
    -rw-rw-r--. 1 simon simon 260 Oct 18 20:44 GFlagsSource.ini
    [simon@localhost opticks]$ ll /usr/local/opticks/installcache/RNG/
    total 129352
    -rw-rw-r--. 1 simon simon    450560 Oct 18 20:44 cuRANDWrapper_10240_0_0.bin
    -rw-rw-r--. 1 simon simon 132000000 Oct 18 20:44 cuRANDWrapper_3000000_0_0.bin
    [simon@localhost opticks]$ 




opticks-prepare-installcache requires completed opticks-full
---------------------------------------------------------------

::

     334 opticks-full()
     335 {
     336     local msg="=== $FUNCNAME :"
     337     echo $msg START $(date)
     338     opticks-info
     339 
     340     if [ ! -d "$(opticks-prefix)/externals" ]; then
     341 
     342         echo $msg installing the below externals into $(opticks-prefix)/externals
     343         opticks-externals
     344         opticks-externals-install
     345 
     346 
     347     else
     348         echo $msg using preexisting externals from $(opticks-prefix)/externals
     349     fi
     350 
     351     opticks-configure
     352 
     353     opticks--
     354 
     355     opticks-prepare-installcache
     356 
     357     echo $msg DONE $(date)
     358 }




SG : ORayleighTest + OpSeederTest + OKTest + VizTest 
------------------------------------------------------

* fails from missing RNG cache



ORayleighTest::

    2017-10-18 20:19:15.798 INFO  [30802] [SLog::operator@15] OScene::OScene DONE
    2017-10-18 20:19:15.798 INFO  [30802] [main@69]  ok 
    cuRANDWrapper::LoadIntoHostBuffer MISSING RNG CACHE AT : /usr/local/opticks/installcache/RNG/cuRANDWrapper_3000000_0_0.bin 
    cuRANDWrapper::LoadIntoHostBuffer : CREATE CACHE WITH bash functions : cudarap-;cudarap-prepare-installcache 
    cuRANDWrapper::LoadIntoHostBuffer : NB cudarap-prepare-installcache SHOULD HAVE BEEN INVOKED BY opticks-prepare-installcache  
    ORayleighTest: /home/simon/opticks/cudarap/cuRANDWrapper.cc:342: int cuRANDWrapper::LoadIntoHostBuffer(curandState*, unsigned int): Assertion `0' failed.

    Program received signal SIGABRT, Aborted.
    0x000000356a432925 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install glibc-2.12-1.132.el6.x86_64 keyutils-libs-1.4-4.el6.x86_64 keyutils-libs-1.4-5.el6.x86_64 krb5-libs-1.10.3-10.el6_4.6.x86_64 krb5-libs-1.10.3-65.el6.x86_64 libcom_err-1.41.12-18.el6.x86_64 libcom_err-1.41.12-23.el6.x86_64 libgcc-4.4.7-17.el6.x86_64 libgcc-4.4.7-18.el6.x86_64 libselinux-2.0.94-5.3.el6_4.1.x86_64 libselinux-2.0.94-7.el6.x86_64 libstdc++-4.4.7-17.el6.x86_64 libstdc++-4.4.7-18.el6.x86_64 openssl-1.0.1e-57.el6.x86_64 zlib-1.2.3-29.el6.x86_64
    (gdb) bt
    #0  0x000000356a432925 in raise () from /lib64/libc.so.6
    #1  0x000000356a434105 in abort () from /lib64/libc.so.6
    #2  0x000000356a42ba4e in __assert_fail_base () from /lib64/libc.so.6
    #3  0x000000356a42bb10 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff29acd36 in cuRANDWrapper::LoadIntoHostBuffer (this=0x5663f20, host_rng_states=0x7fffcba9d010, elements=3000000) at /home/simon/opticks/cudarap/cuRANDWrapper.cc:342
    #5  0x00007ffff29ad481 in cuRANDWrapper::fillHostBuffer (this=0x5663f20, host_rng_states=0x7fffcba9d010, elements=3000000) at /home/simon/opticks/cudarap/cuRANDWrapper.cc:514
    #6  0x00007ffff20dd626 in ORng::init (this=0x7fffffffdd90) at /home/simon/opticks/optixrap/ORng.cc:57
    #7  0x00007ffff20dd257 in ORng::ORng (this=0x7fffffffdd90, ok=0x7fffffffdec0, ocontext=0x1a35b00) at /home/simon/opticks/optixrap/ORng.cc:23
    #8  0x0000000000405fc0 in main (argc=1, argv=0x7fffffffe2d8) at /home/simon/opticks/optixrap/tests/ORayleighTest.cc:72
    (gdb) 


OpSeederTest::

    2017-10-18 20:31:12.733 INFO  [31374] [SLog::operator@15] OScene::OScene DONE
    2017-10-18 20:31:12.733 INFO  [31374] [SLog::operator@15] OEvent::OEvent DONE
    2017-10-18 20:31:12.733 FATAL [31374] [OContext::addEntry@45] OContext::addEntry T
    cuRANDWrapper::LoadIntoHostBuffer MISSING RNG CACHE AT : /usr/local/opticks/installcache/RNG/cuRANDWrapper_3000000_0_0.bin 
    cuRANDWrapper::LoadIntoHostBuffer : CREATE CACHE WITH bash functions : cudarap-;cudarap-prepare-installcache 
    cuRANDWrapper::LoadIntoHostBuffer : NB cudarap-prepare-installcache SHOULD HAVE BEEN INVOKED BY opticks-prepare-installcache  
    OpSeederTest: /home/simon/opticks/cudarap/cuRANDWrapper.cc:342: int cuRANDWrapper::LoadIntoHostBuffer(curandState*, unsigned int): Assertion `0' failed.
    Aborted
    [simon@localhost opticks]$ 


OKTest::

    2017-10-18 20:33:02.563 INFO  [31404] [SLog::operator@15] OScene::OScene DONE
    2017-10-18 20:33:02.563 WARN  [31404] [OpEngine::init@69] OpEngine::init initPropagation START
    2017-10-18 20:33:02.563 FATAL [31404] [OContext::addEntry@45] OContext::addEntry G
    2017-10-18 20:33:02.563 INFO  [31404] [SLog::operator@15] OEvent::OEvent DONE
    cuRANDWrapper::LoadIntoHostBuffer MISSING RNG CACHE AT : /usr/local/opticks/installcache/RNG/cuRANDWrapper_3000000_0_0.bin 
    cuRANDWrapper::LoadIntoHostBuffer : CREATE CACHE WITH bash functions : cudarap-;cudarap-prepare-installcache 
    cuRANDWrapper::LoadIntoHostBuffer : NB cudarap-prepare-installcache SHOULD HAVE BEEN INVOKED BY opticks-prepare-installcache  
    OKTest: /home/simon/opticks/cudarap/cuRANDWrapper.cc:342: int cuRANDWrapper::LoadIntoHostBuffer(curandState*, unsigned int): Assertion `0' failed.
    Aborted

VizTest::


    2017-10-18 20:34:16.093 INFO  [31425] [SLog::operator@15] OScene::OScene DONE
    2017-10-18 20:34:16.093 INFO  [31425] [SLog::operator@15] OEvent::OEvent DONE
    2017-10-18 20:34:16.093 FATAL [31425] [OContext::addEntry@45] OContext::addEntry G
    cuRANDWrapper::LoadIntoHostBuffer MISSING RNG CACHE AT : /usr/local/opticks/installcache/RNG/cuRANDWrapper_3000000_0_0.bin 
    cuRANDWrapper::LoadIntoHostBuffer : CREATE CACHE WITH bash functions : cudarap-;cudarap-prepare-installcache 
    cuRANDWrapper::LoadIntoHostBuffer : NB cudarap-prepare-installcache SHOULD HAVE BEEN INVOKED BY opticks-prepare-installcache  
    VizTest: /home/simon/opticks/cudarap/cuRANDWrapper.cc:342: int cuRANDWrapper::LoadIntoHostBuffer(curandState*, unsigned int): Assertion `0' failed.
    Aborted







