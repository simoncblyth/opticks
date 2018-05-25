opticks-cmake-overhaul-test-fails
==================================

integrated rerun 2/299 fails (now 1/299) (after symbolic links and g4 data reinstall)
---------------------------------------------------------------------------------------

::

   opticks-t /usr/local/opticks-cmake-overhaul-tmp/build

::

    99% tests passed, 2 tests failed out of 299

    Total Test time (real) = 174.98 sec

    The following tests FAILED:
        159 - OpticksCoreTest.OpticksFlagsTest (SEGFAULT)
        222 - GGeoTest.GSceneTest (Child aborted)
    Errors while running CTest
    Fri May 25 16:30:11 HKT 2018
    === opticks-t- : use -V to show output, ctest output written to /usr/local/opticks-cmake-overhaul-tmp/build/ctest.log
    epsilon:issues blyth$ 


Huh OpticksFlagsTest looks fine, and repeating the SEGFAULT goes away::

    epsilon:issues blyth$ /usr/local/opticks-cmake-overhaul-tmp/lib/OpticksFlagsTest
     i   0 flag          1 hs            CERENKOV  hs2             CERENKOV
     i   1 flag          2 hs       SCINTILLATION  hs2        SCINTILLATION
     i   2 flag          4 hs                MISS  hs2                 MISS
     i   3 flag          8 hs         BULK_ABSORB  hs2          BULK_ABSORB
     i   4 flag         16 hs         BULK_REEMIT  hs2          BULK_REEMIT
     i   5 flag         32 hs        BULK_SCATTER  hs2         BULK_SCATTER
     i   6 flag         64 hs      SURFACE_DETECT  hs2       SURFACE_DETECT
     i   7 flag        128 hs      SURFACE_ABSORB  hs2       SURFACE_ABSORB
     i   8 flag        256 hs    SURFACE_DREFLECT  hs2     SURFACE_DREFLECT
     i   9 flag        512 hs    SURFACE_SREFLECT  hs2     SURFACE_SREFLECT
     i  10 flag       1024 hs    BOUNDARY_REFLECT  hs2     BOUNDARY_REFLECT
     i  11 flag       2048 hs   BOUNDARY_TRANSMIT  hs2    BOUNDARY_TRANSMIT
     i  12 flag       4096 hs               TORCH  hs2                TORCH
     i  13 flag       8192 hs           NAN_ABORT  hs2            NAN_ABORT
     i  14 flag      16384 hs               G4GUN  hs2                G4GUN
     i  15 flag      32768 hs          FABRICATED  hs2           FABRICATED
     i  16 flag      65536 hs             NATURAL  hs2              NATURAL
     i  17 flag     131072 hs           MACHINERY  hs2            MACHINERY
     i  18 flag     262144 hs          EMITSOURCE  hs2           EMITSOURCE
     i  19 flag     524288 hs                      hs2             BAD_FLAG
     i  20 flag    1048576 hs                      hs2             BAD_FLAG
     i  21 flag    2097152 hs                      hs2             BAD_FLAG
     i  22 flag    4194304 hs                      hs2             BAD_FLAG
     i  23 flag    8388608 hs                      hs2             BAD_FLAG
     i  24 flag   16777216 hs                      hs2             BAD_FLAG
     i  25 flag   33554432 hs                      hs2             BAD_FLAG
     i  26 flag   67108864 hs                      hs2             BAD_FLAG
     i  27 flag  134217728 hs                      hs2             BAD_FLAG
     i  28 flag  268435456 hs                      hs2             BAD_FLAG
     i  29 flag  536870912 hs                      hs2             BAD_FLAG
     i  30 flag 1073741824 hs                      hs2             BAD_FLAG
     i  31 flag 2147483648 hs                      hs2             BAD_FLAG
    epsilon:issues blyth$ echo $?
    0

    opticks-t /usr/local/opticks-cmake-overhaul-tmp/build/optickscore




integrated 9/299 fails (down to 6/299 after symbolic links under prefix)
----------------------------------------------------------------------------

::

    epsilon:examples blyth$ cat /usr/local/opticks-cmake-overhaul-tmp/build/ctest.log 
    Fri May 25 14:36:04 HKT 2018

    97% tests passed, 9 tests failed out of 299

    Total Test time (real) = 121.45 sec

    The following tests FAILED:
        222 - GGeoTest.GSceneTest (Child aborted)

        265 - OptiXRapTest.ORayleighTest (Child aborted)
        269 - OKOPTest.OpSeederTest (Child aborted)
        276 - OKTest.OKTest (Child aborted)

        282 - CFG4Test.CTestDetectorTest (Child aborted)
        285 - CFG4Test.CG4Test (Child aborted)
        293 - CFG4Test.CInterpolationTest (Child aborted)
        298 - CFG4Test.CRandomEngineTest (Child aborted)

        299 - OKG4Test.OKG4Test (Child aborted)
    Errors while running CTest
    Fri May 25 14:38:05 HKT 2018
    epsilon:examples blyth$ 

::

    /usr/local/opticks-cmake-overhaul-tmp/lib/ORayleighTest 
          MISSING RNG CACHE AT : /usr/local/opticks-cmake-overhaul-tmp/installcache/RNG/cuRANDWrapper_3000000_0_0.bin 

    /usr/local/opticks-cmake-overhaul-tmp/lib/OpSeederTest
    /usr/local/opticks-cmake-overhaul-tmp/lib/OKTest


Symbolic links under the prefix fix 3 of those fails::

    epsilon:installcache blyth$ l ../../opticks-cmake-overhaul/installcache/
    total 0
    drwxr-xr-x  30 blyth  staff  - 960 May 24 17:02 PTX
    drwxr-xr-x   6 blyth  staff  - 192 May 24 13:44 OKC
    drwxr-xr-x   4 blyth  staff  - 128 May 23 19:31 RNG
    epsilon:installcache blyth$ ln -s ../../opticks-cmake-overhaul/installcache/RNG 
    epsilon:installcache blyth$ pwd
    /usr/local/opticks-cmake-overhaul-tmp/installcache
    epsilon:installcache blyth$ ln -s ../../opticks-cmake-overhaul/installcache/OKC
    epsilon:installcache blyth$ 



proj-by-proj  6/299 fails : 5 are G4 related (flags or g4data?) and 1 is an ancient one
-------------------------------------------------------------------------------------------

::

    epsilon:issues blyth$ grep -H passed  /usr/local/opticks-cmake-overhaul/build/*/ctest.log
    /usr/local/opticks-cmake-overhaul/build/assimprap/ctest.log:   100% tests passed, 0 tests failed out of 3
    /usr/local/opticks-cmake-overhaul/build/boostrap/ctest.log:    100% tests passed, 0 tests failed out of 27
    /usr/local/opticks-cmake-overhaul/build/cfg4/ctest.log:         79% tests passed, 4 tests failed out of 19
    /usr/local/opticks-cmake-overhaul/build/cudarap/ctest.log:     100% tests passed, 0 tests failed out of 4
    /usr/local/opticks-cmake-overhaul/build/ggeo/ctest.log:         98% tests passed, 1 tests failed out of 49
    /usr/local/opticks-cmake-overhaul/build/npy/ctest.log:         100% tests passed, 0 tests failed out of 102
    /usr/local/opticks-cmake-overhaul/build/oglrap/ctest.log:      100% tests passed, 0 tests failed out of 2
    /usr/local/opticks-cmake-overhaul/build/ok/ctest.log:          100% tests passed, 0 tests failed out of 5
    /usr/local/opticks-cmake-overhaul/build/okconf/ctest.log:      100% tests passed, 0 tests failed out of 1
    /usr/local/opticks-cmake-overhaul/build/okg4/ctest.log:          0% tests passed, 1 tests failed out of 1
    /usr/local/opticks-cmake-overhaul/build/okop/ctest.log:        100% tests passed, 0 tests failed out of 5
    /usr/local/opticks-cmake-overhaul/build/openmeshrap/ctest.log: 100% tests passed, 0 tests failed out of 1
    /usr/local/opticks-cmake-overhaul/build/optickscore/ctest.log: 100% tests passed, 0 tests failed out of 22
    /usr/local/opticks-cmake-overhaul/build/opticksgeo/ctest.log:  100% tests passed, 0 tests failed out of 3
    /usr/local/opticks-cmake-overhaul/build/optixrap/ctest.log:    100% tests passed, 0 tests failed out of 18
    /usr/local/opticks-cmake-overhaul/build/sysrap/ctest.log:      100% tests passed, 0 tests failed out of 22
    /usr/local/opticks-cmake-overhaul/build/thrustrap/ctest.log:   100% tests passed, 0 tests failed out of 15
    epsilon:issues blyth$ 

    echo $(( 3 + 27 + 19 + 4 + 49 + 102 + 2 + 5 + 1 + 1 + 5 + 1 + 22 + 3 + 18 + 22 + 15 ))
    299




Five G4 fails : all from missing data
----------------------------------------


::

    epsilon:issues blyth$ which CRandomEngineTest
    /usr/local/opticks-cmake-overhaul/lib/CRandomEngineTest
    epsilon:issues blyth$ CRandomEngineTest
    2018-05-25 15:36:40.803 INFO  [8624853] [main@72] CRandomEngineTest
    2018-05-25 15:36:40.805 INFO  [8624853] [main@76]  pindex 0
      0 : CRandomEngineTest

    -------- EEEE ------- G4Exception-START -------- EEEE -------
    *** G4Exception : PART70001
          issued by : G4NuclideTable
    ENSDFSTATE.dat is not found.
    *** Fatal Exception *** core dump ***
    -------- EEEE -------- G4Exception-END --------- EEEE -------


    *** G4Exception: Aborting execution ***
    Abort trap: 6
    epsilon:issues blyth$ 




