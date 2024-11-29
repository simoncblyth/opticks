more_flexible_maxphoton_plus_setting_it_based_on_VRAM_plus_splitting_launch_when_VRAM_too_small_for_photon_count
==================================================================================================================


more_flexible_maxphoton
-------------------------

* work on this in::

     ~/o/sysrap/SCurandState.h 
     ~/o/sysrap/tests/SCurandState_test.sh  


* currently the maxphoton values that can be used depend on the SCurandState files that have been generated
  and those files are very repetitive and large 

* TODO : use smaller files and concatenate the appropriate number for the 
  desired maxphoton, avoiding duplication 

* NP has concat capabilities, use techniques from there to implement concat for SCurandState 




VRAM detection
-----------------

Do that at initialization just before loading states, 
sdevice 



* cuda has device API : ~/o/sysrap/sdevice.h  uses that 
* nvml has C api : ~/o/sysrap/smonitor.{sh,cc} uses that 


Setting maxphoton based on VRAM
--------------------------------


splitting launch to handle more photon that fit into VRAM
--------------------------------------------------------------


