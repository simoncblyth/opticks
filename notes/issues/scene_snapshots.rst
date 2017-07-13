Scene Snapshots
===================



op route
-----------

::

    309 op-geometry-query-dyb()
    310 {
    311     case $1 in
    312    DYB|DLIN)  echo "range:3153:12221"  ;;
    313        DFAR)  echo "range:4686:18894"   ;;  #  
    314        IDYB)  echo "range:3158:3160" ;;  # 2 volumes : pvIAV and pvGDS
    315        JDYB)  echo "range:3158:3159" ;;  # 1 volume : pvIAV
    316        KDYB)  echo "range:3159:3160" ;;  # 1 volume : pvGDS
    317        LDYB)  echo "range:3156:3157" ;;  # 1 volume : pvOAV
    318        MDYB)  echo "range:3201:3202,range:3153:3154"  ;;  # 2 volumes : first pmt-hemi-cathode and ADE  
    319        DSST2)  echo "range:3155:3156,range:4440:4448" ;;    # large BBox discrep
    320        DLV17)  echo "range:3155:3156,range:2436:2437" ;;    #
    321        DLV30)  echo "range:3155:3156,range:3167:3168" ;;    #



    op --idyb --gltf 3  

    op --drv3155 --gltf 3  
         hmm although this applies recursive select to a volume and contents, 
         instancing kicks in so other instanced volumes show up elsewhere 



Attempt to honour_selection in GScene
----------------------------------------

* made this work by moving m_selected down into GNode
* introducing  m_honour_selection in GScene
* allowing getMergedMesh to return NULLs that are just skipped 

::

     913 
     914          const std::vector<GNode*>& instances = m_root->findAllInstances(ridx, inside, m_honour_selection );
     915 



::

    2017-07-13 15:41:33.848 INFO  [5243988] [GScene::init@163] GScene::init createVolumeTrue selected_count 1375
    2017-07-13 15:41:33.882 INFO  [5243988] [GScene::makeMergedMeshAndInstancedBuffers@903] GScene::makeMergedMeshAndInstancedBuffers num_repeats 56 START 
    2017-07-13 15:41:34.123 WARN  [5243988] [GScene::makeMergedMeshAndInstancedBuffers@919] GScene::makeMergedMeshAndInstancedBuffers no instances with ridx 1
    2017-07-13 15:41:34.123 WARN  [5243988] [GScene::makeMergedMeshAndInstancedBuffers@919] GScene::makeMergedMeshAndInstancedBuffers no instances with ridx 2
    2017-07-13 15:41:34.124 WARN  [5243988] [GScene::makeMergedMeshAndInstancedBuffers@919] GScene::makeMergedMeshAndInstancedBuffers no instances with ridx 3
    2017-07-13 15:41:34.127 WARN  [5243988] [GScene::makeMergedMeshAndInstancedBuffers@919] GScene::makeMergedMeshAndInstancedBuffers no instances with ridx 6
    ...
    2017-07-13 15:41:34.153 WARN  [5243988] [GScene::makeMergedMeshAndInstancedBuffers@919] GScene::makeMergedMeshAndInstancedBuffers no instances with ridx 47
    2017-07-13 15:41:34.153 WARN  [5243988] [GScene::makeMergedMeshAndInstancedBuffers@919] GScene::makeMergedMeshAndInstancedBuffers no instances with ridx 48
    2017-07-13 15:41:34.153 WARN  [5243988] [GScene::makeMergedMeshAndInstancedBuffers@919] GScene::makeMergedMeshAndInstancedBuffers no instances with ridx 49
    2017-07-13 15:41:34.154 WARN  [5243988] [GScene::makeMergedMeshAndInstancedBuffers@919] GScene::makeMergedMeshAndInstancedBuffers no instances with ridx 50
    2017-07-13 15:41:34.154 WARN  [5243988] [GScene::makeMergedMeshAndInstancedBuffers@919] GScene::makeMergedMeshAndInstancedBuffers no instances with ridx 51


    2017-07-13 15:41:34.158 WARN  [5243988] [GScene::makeMergedMeshAndInstancedBuffers@919] GScene::makeMergedMeshAndInstancedBuffers no instances with ridx 55
    2017-07-13 15:41:34.158 INFO  [5243988] [GScene::makeMergedMeshAndInstancedBuffers@957] GScene::makeMergedMeshAndInstancedBuffers DONE num_repeats 56 nmm_created 22 nmm 22
    2017-07-13 15:41:34.158 INFO  [5243988] [GScene::checkMergedMeshes@983] GScene::checkMergedMeshes nmm 22 mia 18
    Assertion failed: (mia == 0), function checkMergedMeshes, file /Users/blyth/opticks/ggeo/GScene.cc, line 988.
    /Users/blyth/opticks/bin/op.sh: line 652: 41944 Abort trap: 6           /usr/local/opticks/lib/OKTest --drv3155 --gltf 3
    /Users/blyth/opticks/bin/op.sh RC 134
    simon:ggeo blyth$ 


