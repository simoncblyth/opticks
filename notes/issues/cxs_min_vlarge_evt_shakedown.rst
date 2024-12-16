cxs_min_vlarge_evt_shakedown
================================


::

    TEST=vlarge_evt ~/o/cxs_min.sh 

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





