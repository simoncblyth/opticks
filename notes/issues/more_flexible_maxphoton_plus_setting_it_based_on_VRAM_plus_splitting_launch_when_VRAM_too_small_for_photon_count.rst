more_flexible_maxphoton_plus_setting_it_based_on_VRAM_plus_splitting_launch_when_VRAM_too_small_for_photon_count
==================================================================================================================


more_flexible_maxphoton
-------------------------

* work on this in::

     ~/o/sysrap/SCurandState.h 
     ~/o/sysrap/tests/SCurandState_test.sh  


* PREVIOUSLY : the maxphoton values that can be used depend on the SCurandState files that have been generated
  and those files are very repetitive and large 

* DONE : use chunk files and concatenate the appropriate number for the 
  desired maxphoton, avoiding duplication 

* DONE : also do partial reads on the last chunk to decouple file sizes from maxphoton

* DONE : comparisions at 3M level between old and new using QRngTest.sh match perfectly 

* TODO : large scale test 10M, 100M, 200M 

* TODO : bash level use of the new functionality qudarap-prepare-installation 

* TODO : implement OPTICKS_MAX_PHOTON:0 to correspond to maximum permitted by available
         VRAM and modulo the limitation from the available chunks  

         * give warning when the VRAM is large enough to warrant larger launches
           than the chunks permit 


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

Do that at initialization just before loading states, sdevice is already in use somewhere, 
mainly for metadata purposes. Maybe will need to move it earlier for this purpose. 

* cuda has device API : ~/o/sysrap/sdevice.h  uses that 
* nvml has C api : ~/o/sysrap/smonitor.{sh,cc} uses that 


Setting maxphoton based on VRAM
--------------------------------



splitting launch to handle more photon that fit into VRAM
--------------------------------------------------------------

* doing all the launches at EndOfEvent ? or doing throughout event ? 
* at the launch boundary : is splitting genstep possible (cloning and changing photon count on the excess) ? 
* how to handle SEvt ?



