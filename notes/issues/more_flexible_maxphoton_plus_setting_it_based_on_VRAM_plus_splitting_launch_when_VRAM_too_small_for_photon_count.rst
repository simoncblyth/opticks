more_flexible_maxphoton_plus_setting_it_based_on_VRAM_plus_splitting_launch_when_VRAM_too_small_for_photon_count
==================================================================================================================


more_flexible_maxphoton
-------------------------

* work on this in::

     ~/o/sysrap/SCurandState.h 
     ~/o/sysrap/tests/SCurandState_test.sh  


* currently the maxphoton values that can be used depend on the SCurandState files that have been generated
  and those files are very repetitive and large 

* WIP : use smaller files and concatenate the appropriate number for the 
  desired maxphoton, avoiding duplication 

* WIP: also do partial reads to the nearest million states to decouple file
  sizes from maxphoton




::

    P[blyth@localhost opticks]$ o
    On branch master
    Your branch is up to date with 'origin/master'.

    Changes not staged for commit:
      (use "git add <file>..." to update what will be committed)
      (use "git restore <file>..." to discard changes in working directory)
        modified:   notes/issues/more_flexible_maxphoton_plus_setting_it_based_on_VRAM_plus_splitting_launch_when_VRAM_too_small_for_photon_count.rst
        modified:   qudarap/QCurandState.cc
        modified:   qudarap/QCurandState.hh
        modified:   qudarap/QRng.cc
        modified:   qudarap/tests/QCurandStateTest.cc
        modified:   sysrap/SCurandState.cc
        modified:   sysrap/SCurandState.h
        modified:   sysrap/sdirectory.h
        modified:   sysrap/tests/SCurandState_test.cc
        modified:   sysrap/tests/SCurandState_test.sh
        modified:   sysrap/tests/SLaunchSequence_test.cc
        modified:   sysrap/tests/sdirectory_test.cc

    Untracked files:
      (use "git add <file>..." to include in what will be committed)
        sysrap/tests/SLaunchSequence_test.sh
        sysrap/tests/sdirectory_test.sh




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


