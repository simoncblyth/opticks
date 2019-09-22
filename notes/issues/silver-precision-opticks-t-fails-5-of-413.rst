silver-precision-opticks-t-fails-5-of-413
=============================================


Issue : unexpected opticks-t fails + non-fails + rare ctest/script machinery breakage
-------------------------------------------------------------------------------------------

Unexpected non-failures:

* legacy not enabled, but dont see the slow CG4 OKG4 ?

Next
-----

* :doc:`ctest-running-broken-by-some-types-of-test-fails`


Symptom of ctest breakage : mixed dates on the ctest.log
-----------------------------------------------------------

* YEP : mixed dates, somehow test running is broken as a result of missing rngcache ? 
* perhaps a hang ?


::

    FAILS:  5   / 413   :  Sat Sep 21 23:19:30 2019   
      78 /121 Test #78 : NPYTest.NLoadTest                             ***Exception: SegFault         0.15   
      47 /53  Test #47 : GGeoTest.GPropertyTest                        ***Exception: SegFault         0.10   

             missing opticksaux

      18 /25  Test #18 : OptiXRapTest.interpolationTest                ***Failed                      6.11   

             PATH


      12 /25  Test #12 : OptiXRapTest.rayleighTest                     Child aborted***Exception:     45.58  
      19 /25  Test #19 : OptiXRapTest.ORngTest                         Child aborted***Exception:     1.91   

             missing rngcache : huh should be many more fails ?

::

    (base) [blyth@gilda03 opticks]$ eo
    OPTICKS_EVENT_BASE=/home/blyth/local/opticks/tmp
    OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce
    OPTICKS_COMPUTE_CAPABILITY=70
    OPTICKS_NVIDIA_DRIVER_VERSION=435.21
    OPTICKS_HOME=/home/blyth/opticks
    OPTICKS_ANA_DEFAULTS=det=g4live,cat=cvd_1_rtx_1_1M,src=torch,tag=1,pfx=scan-ph
    OPTICKS_DEFAULT_INTEROP_CVD=0
    OPTICKS_RESULTS_PREFIX=/home/blyth/local/opticks



repeat after opticksaux--
---------------------------------

::

    FAILS:  7   / 416   :  Sun Sep 22 00:17:23 2019   
      18 /25  Test #18 : OptiXRapTest.interpolationTest                ***Failed                      6.43   
             this fail is expected when hopping between geometries

      12 /25  Test #12 : OptiXRapTest.rayleighTest                     Child aborted***Exception:     44.34  
      19 /25  Test #19 : OptiXRapTest.ORngTest                         Child aborted***Exception:     1.95   
      2  /5   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     42.69  
      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     45.37  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     66.55  
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      6.90   
             all these from lack of rngcache, 
             integration-t appears different as run via bash script not directly the executable

    (base) [blyth@gilda03 opticks]$ 



opticks-t should be checking existance of the RNG dir
--------------------------------------------------------

* added checking in opticks-t opticks-check-installation and cudarap-check-installation

::

    (base) [blyth@gilda03 integration]$ l /home/blyth/local/opticks/installcache/
    total 4
    drwxrwxr-x. 3 blyth blyth 4096 Sep 21 22:53 PTX
    drwxrwxr-x. 2 blyth blyth  155 Sep 12 14:03 RNG
    drwxrwxr-x. 2 blyth blyth  143 Sep  9 21:34 OKC
    (base) [blyth@gilda03 integration]$ 

    (base) [blyth@gilda03 integration]$ cd /home/blyth/local/opticks/installcache/    ## RNG is moved, OKC eliminated
    (base) [blyth@gilda03 installcache]$ rm -rf RNG
    (base) [blyth@gilda03 installcache]$ rm -rf OKC

DONE::

    (base) [blyth@gilda03 opticks]$ opticks-t
    cudarap-check-rng- : ERROR rngdir missing /home/blyth/.opticks/rngcache/RNG rc 201
    cudarap-check-rng- : ERROR rngdir missing /home/blyth/.opticks/rngcache/RNG rc 201
    cudarap-check-rng- : ERROR rngdir missing /home/blyth/.opticks/rngcache/RNG rc 201
    === opticks-check-installation : rc 201
    === opticks-t- : ABORT : missing installcache components : create with opticks-prepare-installation
    (base) [blyth@gilda03 opticks]$ 




opticksaux new external needed, this is replacing opticksdata
--------------------------------------------------------------------

::

    (base) [blyth@gilda03 ~]$ opticksaux-
    (base) [blyth@gilda03 ~]$ opticksaux--
    opticksaux-get : proceeding with "git clone https://bitbucket.org/simoncblyth/opticksaux.git " from /home/blyth/local/opticks
    Cloning into 'opticksaux'...
    remote: Counting objects: 28, done.
    remote: Compressing objects: 100% (16/16), done.
    remote: Total 28 (delta 0), reused 0 (delta 0)
    Unpacking objects: 100% (28/28), done.
    (base) [blyth@gilda03 ~]$ 




interpolationTest : silver missing PATH addition to find interpolationTest_interpol.py
-----------------------------------------------------------------------------------------------

::

    $OPTICKS_INSTALL_PREFIX/bin    ## now needed in PATH 


* TODO: standardize such environment setup

::

    2019-09-21 23:23:43.274 INFO  [90795] [interpolationTest::launch@165]  save  base $TMP/interpolationTest name interpolationTest_interpol.npy
    which: no interpolationTest_interpol.py in (/home/blyth/env/bin:/home/blyth/anaconda2/bin:/home/blyth/anaconda2/condabin:/home/blyth/opticks/bin:/home/blyth/opticks/ana:/home/blyth/anaconda2/bin:/home/blyth/.cargo/bin:/home/blyth/local/opticks/lib:/home/blyth/local/bin:/usr/local/cuda-10.1/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin:/home/blyth/.local/bin:/home/blyth/bin)
    2019-09-21 23:23:43.323 INFO  [90795] [interpolationTest::ana@179]  m_script interpolationTest_interpol.py path 
    Python 2.7.16 |Anaconda, Inc.| (default, Mar 14 2019, 21:00:58) 
    [GCC 7.3.0] on linux2
    Type "help", "copyright", "credits" or "license" for more information.
    >>> 
    KeyboardInterrupt
    >>> 
    2019-09-21 23:24:06.587 INFO  [90795] [SSys::run@91] python  rc_raw : 0 rc : 0
    2019-09-21 23:24:06.587 INFO  [90795] [interpolationTest::ana@185]  RC 0
    (base) [blyth@gilda03 opticks]$ 



    (base) [blyth@gilda03 opticks]$ which interpolationTest_interpol.py
    /usr/bin/which: no interpolationTest_interpol.py in (/home/blyth/env/bin:/home/blyth/anaconda2/bin:/home/blyth/anaconda2/condabin:/home/blyth/opticks/bin:/home/blyth/opticks/ana:/home/blyth/anaconda2/bin:/home/blyth/.cargo/bin:/home/blyth/local/opticks/lib:/home/blyth/local/bin:/usr/local/cuda-10.1/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin:/home/blyth/.local/bin:/home/blyth/bin)


    [blyth@localhost ana]$ which interpolationTest_interpol.py
    ~/local/opticks/bin/interpolationTest_interpol.py



rayleighTest
---------------

::

    2019-09-21 23:28:47.290 INFO  [91049] [main@89]  ok 
    2019-09-21 23:28:47.291 ERROR [91049] [cuRANDWrapper::LoadIntoHostBuffer@652]  MISSING RNG CACHE /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_3000000_0_0.bin


