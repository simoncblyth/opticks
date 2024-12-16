cxs_min_vlarge_evt_shakedown
================================


::

    TEST=large_evt ~/o/cxs_min.sh 
    TEST=vlarge_evt ~/o/cxs_min.sh 
    TEST=vvlarge_evt ~/o/cxs_min.sh 

::

    334 elif [ "$TEST" == "vlarge_evt" ]; then
    335 
    336    opticks_num_photon=M500
    337    opticks_num_genstep=20
    338    opticks_num_event=1
    339    opticks_running_mode=SRM_TORCH
    340    #opticks_max_photon=M200        ## G1 default so no need to set  
    341    #opticks_max_slot=0              ## zero -> SEventConfig::SetDevice determines MaxSlot based on VRAM   
    342 

    343 elif [ "$TEST" == "vvlarge_evt" ]; then
    344 
    345    opticks_num_photon=G1
    346    opticks_num_genstep=40
    347    opticks_num_event=1
    348    opticks_running_mode=SRM_TORCH


Comment the max Philox defaults cover it::

      64 #if defined(RNG_XORWOW)
      65 const char* SEventConfig::_MaxCurandDefault = "M3" ; 
      66 const char* SEventConfig::_MaxSlotDefault = "M3" ;  
      67 const char* SEventConfig::_MaxGenstepDefault = "M3" ; 
      68 const char* SEventConfig::_MaxPhotonDefault = "M3" ;  
      69 const char* SEventConfig::_MaxSimtraceDefault = "M3" ; 
      70 
      71 #elif defined(RNG_PHILOX) || defined(RNG_PHILITEOX)
      72 const char* SEventConfig::_MaxCurandDefault = "G1" ; // nominal 1-billion states, as Philox has no need for curandState loading  
      73 const char* SEventConfig::_MaxSlotDefault = "0" ;     // see SEventConfig::SetDevice : set according to VRAM  
      74 const char* SEventConfig::_MaxGenstepDefault = "M10" ;  // adhoc  
      75 const char* SEventConfig::_MaxPhotonDefault = "G1" ; 
      76 const char* SEventConfig::_MaxSimtraceDefault = "G1" ;
      77 #endif



Ada managed 1 billion photons in 4 launches of 250M taking 2 min clocktime, kernel time less than 100s::

    2024-12-16 14:03:30.575 INFO  [56770] [QSim::simulate@385] sslice {    0,   10,      0,250000000}
    2024-12-16 14:03:56.724 INFO  [56770] [QSim::simulate@385] sslice {   10,   20,250000000,250000000}
    2024-12-16 14:04:23.235 INFO  [56770] [QSim::simulate@385] sslice {   20,   30,500000000,250000000}
    2024-12-16 14:04:49.998 INFO  [56770] [QSim::simulate@385] sslice {   30,   40,750000000,250000000}
    2024-12-16 14:05:29.785 INFO  [56770] [QSim::simulate@423]  eventID 0 tot_dt   94.502935 ph  250000000 ph/M        250 ht  215633111 ht/M        215 reset_ YES
    2024-12-16 14:05:29.785 INFO  [56770] [SEvt::save@3993] /data1/blyth/tmp/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1_Debug_Philox_vvlarge_evt/A000 [genstep,hit]



    2024-12-16 16:23:20.381 INFO  [62044] [QSim::simulate@391] sslice {    0,   10,      0,250000000}
    2024-12-16 16:23:46.508 INFO  [62044] [QSim::simulate@391] sslice {   10,   20,250000000,250000000}
    2024-12-16 16:24:13.038 INFO  [62044] [QSim::simulate@391] sslice {   20,   30,500000000,250000000}
    2024-12-16 16:24:39.798 INFO  [62044] [QSim::simulate@391] sslice {   30,   40,750000000,250000000}
    2024-12-16 16:25:18.928 INFO  [62044] [QSim::simulate@426]  eventID 0 tot_dt   94.236706 tot_ph 1000000000 tot_ph/M       1000 tot_ht  215633111 tot_ht/M        215 last_launch_num_ph  250000000 last_launch_num_ph/M        250 tot_ht/tot_ph          0 reset_ YES
    2024-12-16 16:25:18.929 INFO  [62044] [SEvt::save@3994] /data1/blyth/tmp/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1_Debug_Philox_vvlarge_evt/A000 [genstep,hit]
    2024-12-16 16:27:09.113 INFO  [62044] [QSim::simulate@443] 
    SEvt__MINTIME
     (LEND - LBEG)  106256689 (LEND - LBEG)/M        106 (multilaunch loop begin to end) 
     (PCAT - LEND)   12290570 (PCAT - LEND)/M         12 (topfold concat and clear subfold) 
     (TAIL - HEAD)  228732486 (TAIL - HEAD)/M        228 (head to tail of QSim::simulate method) 
     tot_idt   94236771 tot_idt/M           94       (sum of kernel execution int64_t stamp differences in microseconds)
     tot_dt  94.236706  int64_t(tot_dt*M)   94236705 (sum of kernel execution double chrono stamp differences in seconds, and scaled to ms) 





With 13G of hits::

    A[blyth@localhost CSGOptiX]$ TEST=vvlarge_evt ~/o/cxs_min.sh du
    13G /data1/blyth/tmp/GEOM/J_2024aug27/CSGOptiXSMTest/ALL1_Debug_Philox_vvlarge_evt/A000/hit.npy




Need review of profiling info (probably updates for multi-launch needed), prior to scanning
----------------------------------------------------------------------------------------------










