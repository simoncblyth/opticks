cxs_min_large_evt_shakedown
============================


::

    TEST=large_evt ~/o/cxs_min.sh 



WIP : check the large event on Ada
--------------------------------------

1. DONE : update /cvmfs with GEOM J_2024nov27
2. WIP: update opticks and check opticks-t 

TODO : review all use of max_photon : many of them need to be max_slot
-----------------------------------------------------------------------------

TODO : revisit sevent reporting of the max 
--------------------------------------------



Issue 1 : genstep slice check rng_state assert
--------------------------------------------------

* rngmax messed up by empty string OPTICKS_MAX_CURAND="" leading to max_curand -1

  * changed ssys::getenvvar with empty string value to use fallback 
  * changed SEventConfig::_MaxCurandDefault to "G1" nominal 1 Giga-states 


SEventConfig::

  79 #elif defined(RNG_PHILOX) || defined(RNG_PHILITEOX)
  80 const char* SEventConfig::_MaxCurandDefault = "G1" ;
  81 const char* SEventConfig::_MaxSlotDefault = "M3" ;       // HMM: want to use 0 to signify set it automatically based on VRAM 
  82 const char* SEventConfig::_MaxGenstepDefault = "M3" ;
  83 const char* SEventConfig::_MaxPhotonDefault = "M3" ;
  84 const char* SEventConfig::_MaxSimtraceDefault = "M3" ;
  85 #endif
  86 






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


rngmax
----------

::

    In [1]: np.uint64(-1)
    Out[1]: 18446744073709551615




