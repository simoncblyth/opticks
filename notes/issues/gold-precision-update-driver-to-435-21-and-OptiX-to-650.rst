gold-precision-update-driver-to-435-21-and-OptiX-to-650
============================================================


do the update as noted
--------------------------

::

   onvidia-
   optix-


opticks-t with driver 435.21 and OptiX 650 : 1/415 fails : torus tests passing with Optix 6.5.0
--------------------------------------------------------------------------------------------------

::

    FAILS:  1   / 415   :  Fri Sep 20 14:17:22 2019   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      14.96  


The one fail is a known analysis alignment mismatch for scatters between G4 and Opticks 
when WITH_LOGDOUBLE is not being used, as visible with OpticksSwitchesTest::

    [blyth@localhost issues]$ OpticksSwitchesTest
    2019-09-20 14:34:55.703 INFO  [122424] [main@30] WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_ALIGN_DEV_DEBUG WITH_KLUDGE_FLAT_ZERO_NOPEEK 


OKTest --xanalytic : fails creating analytic geom when legacy is enabled 
----------------------------------------------------------------------------



scan-ph-8
-------------

::

    247 7
    248    Gold:TITAN_RTX 418.56:OptiX 600:WITH_LOGDOUBLE commented:LEGACY_ENABLED:
    249    reproducibility check following other okdist developments before updating driver and OptiX
    250 8
    251    Gold:TITAN_RTX 435.21:OptiX 650:WITH_LOGDOUBLE commented:LEGACY_ENABLED:
    252    after updating driver and OptiX
    253  


::

    [blyth@localhost examples]$ scan-ph-
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_0_1M --generateoverride 1000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 3 --cvd 1 --rtx 0
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_0_10M --generateoverride 10000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 10 --cvd 1 --rtx 0
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_0_20M --generateoverride 20000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 100 --cvd 1 --rtx 0
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_0_30M --generateoverride 30000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 100 --cvd 1 --rtx 0
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_0_40M --generateoverride 40000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 100 --cvd 1 --rtx 0
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_0_50M --generateoverride 50000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 100 --cvd 1 --rtx 0
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_0_60M --generateoverride 60000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 100 --cvd 1 --rtx 0
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_0_70M --generateoverride 70000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 100 --cvd 1 --rtx 0
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_0_80M --generateoverride 80000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 100 --cvd 1 --rtx 0
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_0_90M --generateoverride 90000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 100 --cvd 1 --rtx 0
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_0_100M --generateoverride 100000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 100 --cvd 1 --rtx 0
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_1_1M --generateoverride 1000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 3 --cvd 1 --rtx 1
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_1_10M --generateoverride 10000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 10 --cvd 1 --rtx 1
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_1_20M --generateoverride 20000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 100 --cvd 1 --rtx 1
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_1_30M --generateoverride 30000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 100 --cvd 1 --rtx 1
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_1_40M --generateoverride 40000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 100 --cvd 1 --rtx 1
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_1_50M --generateoverride 50000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 100 --cvd 1 --rtx 1
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_1_60M --generateoverride 60000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 100 --cvd 1 --rtx 1
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_1_70M --generateoverride 70000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 100 --cvd 1 --rtx 1
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_1_80M --generateoverride 80000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 100 --cvd 1 --rtx 1
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_1_90M --generateoverride 90000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 100 --cvd 1 --rtx 1
    ts box --pfx scan-ph-8 --cat cvd_1_rtx_1_100M --generateoverride 100000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 100 --cvd 1 --rtx 1
    [blyth@localhost examples]$ 




flip the WITH_LOGDOUBLE switch 
--------------------------------------

::

    okc-c
    vi OpticksSwitches.hh
    om--

    OpticksSwitchesTest

    o
    om--   ## must rebuild everything, not just okc lib : as the switch is in the header

    integration-t 



Does not fix the fail when forget to recompile everything::

    AB(1,torch,tboolean-box)  None 0     file_photons 10k   load_slice 0:100k:   loaded_photons 10k  
    ab.rpost_dv
    maxdvmax:0.1652  ndvp:  11  level:FATAL  RC:1       skip:
                     :                                :                   :                       :    11    11    11 : 0.0151 0.0220 0.0289 :                                    
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem :  nwar  nerr  nfat :   fwar   ferr   ffat :        mx        mn       avg      
     0000            :                    TO BT BT SA :    8807     8807  :        8807    140912 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0001            :                       TO BR SA :     580      580  :         580      6960 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0002            :                 TO BT BR BT SA :     562      562  :         562     11240 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0003            :              TO BT BR BR BT SA :      29       29  :          29       696 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0004            :           TO BT BR BR BR BT SA :       6        6  :           6       168 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0005            :                       TO SC SA :       5        5  :           5        60 :     5     5     5 : 0.0833 0.0833 0.0833 :    0.0826    0.0413    0.0046   :                FATAL :   > dvmax[2] 0.0289  
     0006            :                       TO BT AB :       2        2  :           2        24 :     2     2     2 : 0.0833 0.0833 0.0833 :    0.0413    0.0413    0.0034   :                FATAL :   > dvmax[2] 0.0289  
     0007            :                 TO BT BT SC SA :       2        2  :           2        40 :     2     2     2 : 0.0500 0.0500 0.0500 :    0.1652    0.0551    0.0055   :                FATAL :   > dvmax[2] 0.0289  
     0008            :              TO SC BT BR BT SA :       1        1  :           1        24 :     1     1     1 : 0.0417 0.0417 0.0417 :    0.0963    0.0963    0.0040   :                FATAL :   > dvmax[2] 0.0289  
     0009            :     TO BT SC BR BR BR BR BT SA :       1        1  :           1        36 :     1     1     1 : 0.0278 0.0278 0.0278 :    0.1239    0.1239    0.0034   :                FATAL :   > dvmax[2] 0.0289  
    .


After doing so it passes::

    [blyth@localhost integration]$ integration-t
    === om-test-one : integration     /home/blyth/opticks/integration                              /home/blyth/local/opticks/build/integration                  
    Fri Sep 20 15:28:37 CST 2019
    Test project /home/blyth/local/opticks/build/integration
        Start 1: IntegrationTests.IntegrationTest
    1/2 Test #1: IntegrationTests.IntegrationTest ...   Passed    0.01 sec
        Start 2: IntegrationTests.tboolean.box
    2/2 Test #2: IntegrationTests.tboolean.box ......   Passed   15.56 sec

    100% tests passed, 0 tests failed out of 2

    Total Test time (real) =  15.58 sec
    Fri Sep 20 15:28:53 CST 2019
    [blyth@localhost integration]$ 



scan-ph-8 : performance check following driver and OptiX update to 6.5.0
---------------------------------------------------------------------------

* small but significant gains in RTX factor : ~6.2x rather than ~6.0x  


scan-ph-9 : check performance WITH_LOGDOUBLE using TITAN RTX
-------------------------------------------------------------------

* by eye the performance looks the same for the full set of plots 


