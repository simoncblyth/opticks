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



Shakedown QRng::LoadAndUpload
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    OPTICKS_MAX_PHOTON=M15  ~/o/qudarap/tests/QRngTest.sh


    QRng::LoadAndUpload rngmax 15000000 rngmax/M 15 available_chunk 24 cs.all.num/M 200 d0 0x7fd346000000
    QRng::LoadAndUpload i 0 chunk.ref.num/M 1 count/M 0 remaining/M 15 partial_read NO  num/M 1 d 0x7fd346000000
    QRng::LoadAndUpload i 1 chunk.ref.num/M 1 count/M 1 remaining/M 14 partial_read NO  num/M 1 d 0x7fd348dc6c00
    QRng::LoadAndUpload i 2 chunk.ref.num/M 1 count/M 2 remaining/M 13 partial_read NO  num/M 1 d 0x7fd34bb8d800
    QRng::LoadAndUpload i 3 chunk.ref.num/M 1 count/M 3 remaining/M 12 partial_read NO  num/M 1 d 0x7fd34e954400
    QRng::LoadAndUpload i 4 chunk.ref.num/M 1 count/M 4 remaining/M 11 partial_read NO  num/M 1 d 0x7fd35171b000
    QRng::LoadAndUpload i 5 chunk.ref.num/M 1 count/M 5 remaining/M 10 partial_read NO  num/M 1 d 0x7fd3544e1c00
    QRng::LoadAndUpload i 6 chunk.ref.num/M 1 count/M 6 remaining/M 9 partial_read NO  num/M 1 d 0x7fd3572a8800
    QRng::LoadAndUpload i 7 chunk.ref.num/M 1 count/M 7 remaining/M 8 partial_read NO  num/M 1 d 0x7fd35a06f400
    QRng::LoadAndUpload i 8 chunk.ref.num/M 1 count/M 8 remaining/M 7 partial_read NO  num/M 1 d 0x7fd35ce36000
    QRng::LoadAndUpload i 9 chunk.ref.num/M 1 count/M 9 remaining/M 6 partial_read NO  num/M 1 d 0x7fd35fbfcc00
    QRng::LoadAndUpload i 10 chunk.ref.num/M 10 count/M 10 remaining/M 5 partial_read YES num/M 5 d 0x7fd3629c3800
    QRngTest: /home/blyth/opticks/qudarap/QRng.cc:287: static curandState* QRng::LoadAndUpload(QRng::ULL, const _SCurandState&): Assertion `cr.num == num' failed.
    /home/blyth/o/qudarap/tests/QRngTest.sh: line 67: 437477 Aborted                 (core dumped) $bin
    /home/blyth/o/qudarap/tests/QRngTest.sh : run error
    P[blyth@localhost qudarap]$ 




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


