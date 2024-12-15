cxs_min_large_evt_shakedown
============================


::

    TEST=large_evt ~/o/cxs_min.sh 



WIP : check large_evt on Ada
--------------------------------------

1. DONE : update /cvmfs with GEOM J_2024nov27
2. DONE: update opticks 
3. DONE : Skip SCurandStateMonolithicTest for RNG_PHILOX
4. WIP: get opticks-t to pass
5. TODO: get ~/o/qudarap/tests/QSimTest_ALL.sh to pass 

::

    A[blyth@localhost ~]$ SCurandStateMonolithicTest
    2024-12-15 22:04:37.604 INFO  [44143] [main@11] 
     spec      1:0:0 num    1000000 seed          0 offset          0 path /home/blyth/.opticks/rngcache/RNG/QCurandStateMonolithic_1M_0_0.bin exists 1 rngmax 1000000
     spec      3:0:0 num    3000000 seed          0 offset          0 path /home/blyth/.opticks/rngcache/RNG/QCurandStateMonolithic_3M_0_0.bin exists 0 rngmax 0
     spec     10:0:0 num   10000000 seed          0 offset          0 path /home/blyth/.opticks/rngcache/RNG/QCurandStateMonolithic_10M_0_0.bin exists 0 rngmax 0
    spath::Filesize unable to open file [/home/blyth/.opticks/rngcache/RNG/QCurandStateMonolithic_3M_0_0.bin]
    SCurandStateMonolithicTest: /home/blyth/opticks/sysrap/spath.h:852: static long int spath::Filesize(const char*): Assertion `!failed' failed.
    Aborted (core dumped)
    A[blyth@localhost ~]$ 


Issue 2 : opticks-t Ada fails
----------------------------------

::

    SLOW: tests taking longer that 15 seconds
      108/109 Test #108: SysRapTest.SSimTest                           Passed                         17.82  
      2  /22  Test #2  : QUDARapTest.QRngTest                          ***Failed                      237.74 


    FAILS:  4   / 218   :  Sun Dec 15 22:31:06 2024   
      2  /22  Test #2  : QUDARapTest.QRngTest                          ***Failed                      237.74 
      10 /22  Test #10 : QUDARapTest.QEventTest                        ***Failed                      0.16   
      11 /22  Test #11 : QUDARapTest.QEvent_Lifecycle_Test             ***Failed                      0.18   
      13 /22  Test #13 : QUDARapTest.QSimWithEventTest                 ***Failed                      2.18   






SKIP : review all use of max_photon : many of them need to be max_slot
-----------------------------------------------------------------------------

Hmm : easier to just set max_photon to very large value 




Issue 1 : genstep slice check rng_state assert
--------------------------------------------------

* rngmax messed up by empty string OPTICKS_MAX_CURAND="" leading to max_curand -1::

    In [1]: np.uint64(-1)
    Out[1]: 18446744073709551615

* changed ssys::getenvvar with empty string value to use fallback 
* changed SEventConfig::_MaxCurandDefault to "G1" nominal 1 Giga-states 

::

    2024-12-15 19:14:20.432  432273926 : [./cxs_min.sh 
    2024-12-15 19:14:22.218 INFO  [68680] [SEventConfig::SetDevice@1295] SEventConfig::DescDevice
    name                             : NVIDIA TITAN RTX
    totalGlobalMem_bytes             : 25396576256
    totalGlobalMem_GB                : 23
    HeuristicMaxSlot(VRAM)           : 197276976
    HeuristicMaxSlot(VRAM)/M         : 197
    HeuristicMaxSlot_Rounded(VRAM)   : 197000000
    MaxSlot/M                        : 0

    2024-12-15 19:14:22.219 INFO  [68680] [SEventConfig::SetDevice@1307]  Configured_MaxSlot/M 0 Final_MaxSlot/M 197 HeuristicMaxSlot_Rounded/M 197 changed YES
     (export OPTICKS_MAX_SLOT=0 # to use VRAM based HeuristicMaxPhoton) 
    2024-12-15 19:14:22.439 INFO  [68680] [QRng::initStates@72] initStates<Philox> DO NOTHING 
    2024-12-15 19:14:22.439 INFO  [68680] [QRng::init@100] [QRng__init_VERBOSE] YES
    QRng::desc
                             IMPL CHUNKED_CURANDSTATE
                          RNGNAME Philox
                UPLOAD_RNG_STATES NO 
                             seed 0
                           offset 0
                           rngmax 18446744073709551615
                         rngmax/M 18446744073709
                               qr 0x13e03580
        qr.skipahead_event_offset 100000
                             d_qr 0x7f3e24600200

    2024-12-15 19:14:22.802 INFO  [68680] [QSim::simulate@385] sslice {    0,    9,      0,180000000}
    2024-12-15 19:14:22.853 FATAL [68680] [QEvent::setGenstepUpload_NP@230]  gss.desc sslice {    0,    9,      0,180000000}
     gss->ph_offset 0
     gss->ph_count 180000000
     gss->ph_offset + gss->ph_count 180000000(last_rng_state_idx) must be <= max_curand for valid rng_state access
     evt->max_curand -1
     evt->num_curand 0
     evt->max_slot 197000000

    CSGOptiXSMTest: /home/blyth/opticks/qudarap/QEvent.cc:241: int QEvent::setGenstepUpload_NP(const NP*, const sslice*): Assertion `in_range' failed.
    ./cxs_min.sh: line 533: 68680 Aborted                 (core dumped) $bin
    ./cxs_min.sh run error
    P[blyth@localhost opticks]$ 




