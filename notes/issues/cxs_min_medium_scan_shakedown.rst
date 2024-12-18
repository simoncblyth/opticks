cxs_min_medium_scan_shakedown
===============================

::

   TEST=medium_scan ~/o/cxs_min.sh 

::

    319 elif [ "$TEST" == "medium_scan" ]; then
    320 
    321    opticks_num_photon=M1,1,10,20,30,40,50,60,70,80,90,100  # duplication of M1 is to workaround lack of metadata
    322    opticks_num_genstep=1,1,1,1,1,1,1,1,1,1,1,1
    323    opticks_max_photon=M100
    324    opticks_num_event=12
    325    opticks_running_mode=SRM_TORCH
    326 


Tedious repetition for num_genstep::

     106 std::vector<int>* SEventConfig::_GetNumGenstepPerEvent()
     107 {
     108     const char* spec = ssys::getenvvar(kNumGenstep,  _NumGenstepDefault );
     109     return sstr::ParseIntSpecList<int>( spec, ',' );
     110 }
     111 std::vector<int>* SEventConfig::_NumGenstepPerEvent = _GetNumGenstepPerEvent() ;
     112 

Added handling of 1x12 to sstr::ParseIntSpecList::

    318 elif [ "$TEST" == "medium_scan" ]; then
    319 
    320    opticks_num_event=12
    321    opticks_num_genstep=1x12
    322    opticks_num_photon=M1,1,10,20,30,40,50,60,70,80,90,100  # duplication of M1 is to workaround lack of metadata
    323    opticks_running_mode=SRM_TORCH
    324    #opticks_max_photon=M100   
    325 




