avoiding_NPFold_save_of_empties_has_consequences_for_Galactic_material_with_no_props
======================================================================================


::

    ################################################################################
    [15:45:09][localhost.localdomain]~/juno-cmake-version-without-qt/opticks % ./cxr_min.sh 
                    GEOM : J23_1_0_rc3_ok0 
                    TMIN : 0.5 
                  LOGDIR : /tmp/ihep/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXRMTest 
                    BASE : /tmp/ihep/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXRMTest 
                    PBAS : /tmp/ihep/opticks/ 
              NAMEPREFIX : cxr_min__eye_1,0,5__zoom_2__tmin_0.5_ 
            OPTICKS_HASH : 246bcc24d 
                 TOPLINE : ESCALE=extent EYE=1,0,5 TMIN=0.5 MOI=NNVT:0:1000 ZOOM=2 CAM=perspective ~/opticks/CSGOptiX/cxr_min.sh  
                 BOTLINE : Fri Jan 19 15:45:24 CST 2024 
                     CVD :  
    CUDA_VISIBLE_DEVICES : 1 
    U::DirList FAILED TO OPEN DIR /home/ihep/.opticks/GEOM/J23_1_0_rc3_ok0/CSGFoundry/SSim/stree/material/Galactic
    ./cxr_min.sh run error
    ####################################################

    I also check this directory:

    [15:51:48][localhost.localdomain]~/juno-cmake-version-without-qt/opticks % ls /home/ihep/.opticks/GEOM/J23_1_0_rc3_ok0/CSGFoundry/SSim/stree/material/
    Acrylic      Air               LatticedShellSteel  NPFold_index.txt  Pyrex  Steel       Tyvek   vetoWater
    AcrylicMask  CDReflectorSteel  LS                  NPFold_names.txt  Rock   StrutSteel  Vacuum  Water

    ##############################

    It seems that the Galactic material is not being saved. I used your jok-tds to generate the CSGFoundary. 
    Is there something I did wrong?



::

    epsilon:np blyth$ git diff f844526e2789220251d6fb7e01cbde409bfac15f 400b7ebf791a9fb4ad1304b29bc95afed9a5af42



Reproduced this in::

    tests/NPFold_save_load_empty_test.sh
    tests/NPFold_save_load_empty_test.cc


    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGINT
      * frame #0: 0x00007fff7d7a1b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff7d96c080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff7d6af6fe libsystem_c.dylib`raise + 26
        frame #3: 0x0000000100065fd0 NPFold_save_load_empty_test`U::DirList(names=size=0, path="/data/blyth/opticks/NPFold_save_load_empty_test_fold/b", ext=0x0000000000000000, exclude=false) at NPU.hh:1391
        frame #4: 0x0000000100063345 NPFold_save_load_empty_test`NPFold::load_dir(this=0x0000000100301900, _base="/data/blyth/opticks/NPFold_save_load_empty_test_fold/b") at NPFold.h:1797
        frame #5: 0x00000001000623ec NPFold_save_load_empty_test`NPFold::load(this=0x0000000100301900, _base="/data/blyth/opticks/NPFold_save_load_empty_test_fold/b") at NPFold.h:1879
        frame #6: 0x0000000100062270 NPFold_save_load_empty_test`NPFold::Load_(base="/data/blyth/opticks/NPFold_save_load_empty_test_fold/b") at NPFold.h:445
        frame #7: 0x000000010008702e NPFold_save_load_empty_test`NPFold::Load(base_="/data/blyth/opticks/NPFold_save_load_empty_test_fold", rel_="b") at NPFold.h:488
        frame #8: 0x00000001000657c5 NPFold_save_load_empty_test`NPFold::load_subfold(this=0x0000000100300e10, _base="/data/blyth/opticks/NPFold_save_load_empty_test_fold", relp="b") at NPFold.h:1713
        frame #9: 0x00000001000631a1 NPFold_save_load_empty_test`NPFold::load_index(this=0x0000000100300e10, _base="/data/blyth/opticks/NPFold_save_load_empty_test_fold") at NPFold.h:1840
        frame #10: 0x00000001000623d7 NPFold_save_load_empty_test`NPFold::load(this=0x0000000100300e10, _base="/data/blyth/opticks/NPFold_save_load_empty_test_fold") at NPFold.h:1879
        frame #11: 0x0000000100062270 NPFold_save_load_empty_test`NPFold::Load_(base="/data/blyth/opticks/NPFold_save_load_empty_test_fold") at NPFold.h:445
        frame #12: 0x000000010002b93c NPFold_save_load_empty_test`NPFold::Load(base_="$FOLD") at NPFold.h:483
        frame #13: 0x000000010002b51c NPFold_save_load_empty_test`main(argc=1, argv=0x00007ffeefbfe930) at NPFold_save_load_empty_test.cc:24
        frame #14: 0x00007fff7d651015 libdyld.dylib`start + 1
    (lldb) 

