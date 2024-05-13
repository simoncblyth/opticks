missing_frame_from_gxt
--------------------------

FIXED: add the ce0 frame even when no framespec 


::

    [blyth@localhost tests]$ SCENE=2 ./ssst.sh 
             BASH_SOURCE : ./ssst.sh
             CUDA_PREFIX : /usr/local/cuda
            OPTIX_PREFIX : /home/blyth/local/opticks/externals/OptiX_750
                  cuda_l : lib64
              SCENE_FOLD : /tmp/blyth/opticks/G4CXOpticks_setGeometry_Test/CSGFoundry/SSim
                    FOLD : /tmp/blyth/opticks/SGLFW_SOPTIX_Scene_test
              SOPTIX_PTX : /tmp/blyth/opticks/SGLFW_SOPTIX_Scene_test/SOPTIX.ptx
                     bin : /tmp/blyth/opticks/SGLFW_SOPTIX_Scene_test/SGLFW_SOPTIX_Scene_test
             SGLFW_FRAME : -1
    ./ssst.sh ptx
    ./ssst.sh ptx DONE
    ./ssst.sh build
    ./ssst.sh build DONE
    ./ssst.sh : Linux running /tmp/blyth/opticks/SGLFW_SOPTIX_Scene_test/SGLFW_SOPTIX_Scene_test : with some manual LD_LIBRARY_PATH config
    NPFold::load non-existing base[/tmp/blyth/opticks/G4CXOpticks_setGeometry_Test/CSGFoundry/SSim/scene/frame]
    SScene::check num_frame 0 num_frame_expect NO 
    SGLFW_SOPTIX_Scene_test: ../SScene.h:125: void SScene::check() const: Assertion `num_frame_expect' failed.
    ./ssst.sh: line 263: 51451 Aborted                 (core dumped) $bin
    ./ssst.sh : run error
    [blyth@localhost tests]$ 
    [blyth@localhost tests]$ 

    [blyth@localhost tests]$ l /tmp/blyth/opticks/G4CXOpticks_setGeometry_Test/CSGFoundry/SSim/scene/
    total 12
    0 -rw-rw-r--. 1 blyth blyth   0 May 13 17:28 NPFold_names.txt
    4 -rw-rw-r--. 1 blyth blyth 144 May 13 17:28 inst_info.npy
    4 -rw-rw-r--. 1 blyth blyth 192 May 13 17:28 inst_tran.npy
    4 -rw-rw-r--. 1 blyth blyth  54 May 13 17:28 NPFold_index.txt
    0 drwxr-xr-x. 4 blyth blyth 130 May 13 16:52 .
    0 drwxr-xr-x. 3 blyth blyth  63 May 13 16:52 meshgroup
    0 drwxr-xr-x. 3 blyth blyth  63 May 13 16:52 meshmerge
    0 drwxr-xr-x. 5 blyth blyth  93 May 13 16:52 ..

    [blyth@localhost tests]$ l /tmp/SScene_test/scene/
    total 16
    0 -rw-rw-r--.  1 blyth blyth    0 May 13 14:15 NPFold_names.txt
    4 -rw-rw-r--.  1 blyth blyth  192 May 13 14:15 inst_tran.npy
    4 -rw-rw-r--.  1 blyth blyth   54 May 13 14:15 NPFold_index.txt
    4 -rw-rw-r--.  1 blyth blyth  144 May 13 14:15 inst_info.npy
    0 drwxrwxr-x.  3 blyth blyth   38 May 13 14:15 ..
    4 drwxr-xr-x.  2 blyth blyth 4096 May 11 17:12 frame
    0 drwxr-xr-x.  5 blyth blyth  143 Apr 10 20:55 .
    0 drwxr-xr-x. 11 blyth blyth  135 Apr 10 19:44 meshgroup
    0 drwxr-xr-x. 11 blyth blyth  135 Apr 10 19:44 meshmerge
    [blyth@localhost tests]$ 



::

     43 inline int SScene_test::CreateFromTree()
     44 {
     45     std::cout << "[SScene_test::CreateFromTree" << std::endl ;
     46     stree* st = stree::Load("$TREE_FOLD");
     47     std::cout << st->desc() ;
     48 
     49     SScene scene ;
     50     scene.initFromTree(st);
     51 
     52     std::cout << scene.desc() ;
     53     scene.save("$SCENE_FOLD") ;  // "scene" reldir is implicit 
     54 
     55     std::cout << "]SScene_test::CreateFromTree" << std::endl ;
     56     return 0 ;
     57 }

    142 inline void SScene::initFromTree(const stree* st)
    143 {
    144     initFromTree_Remainder(st);
    145     initFromTree_Factor(st);
    146     initFromTree_Instance(st);
    147 
    148     addFrames("$SScene__initFromTree_addFrames", st );
    149 }








