Scene Snapshots
===================


tricks
--------

* use Z to back out the zoom
* shift-num to make bookmarks
* U then T to animate between bookmarks



op route
-----------

::

    310 op-geometry-query-dyb()
    311 {
    312     case $1 in
    313    DYB|DLIN)  echo "range:3153:12221"  ;;
    314        DFAR)  echo "range:4686:18894"   ;;  #  
    315        IDYB)  echo "range:3158:3160" ;;  # 2 volumes : pvIAV and pvGDS
    316        JDYB)  echo "range:3158:3159" ;;  # 1 volume : pvIAV
    317        KDYB)  echo "range:3159:3160" ;;  # 1 volume : pvGDS
    318        LDYB)  echo "range:3156:3157" ;;  # 1 volume : pvOAV
    319        MDYB)  echo "range:3201:3202,range:3153:3154"  ;;  # 2 volumes : first pmt-hemi-cathode and ADE  
    320        DSST2)  echo "range:3155:3156,range:4440:4448" ;;    # large BBox discrep
    321        DRV3155) echo "index:3155,depth:20" ;;
    322        DLV17)  echo "range:3155:3156,range:2436:2437" ;;    #
    323        DLV30)  echo "range:3155:3156,range:3167:3168" ;;    #
    324        DLV46)  echo "range:3155:3156,range:3200:3201" ;;    #
    325        DLV55)  echo "range:3155:3156,range:4357:4358" ;;    #
    326        DLV56)  echo "range:3155:3156,range:4393:4394" ;;    #
    327        DLV65)  echo "range:3155:3156,range:4440:4441" ;;
    328        DLV66)  echo "range:3155:3156,range:4448:4449" ;;
    329        DLV67)  echo "range:3155:3156,range:4456:4457" ;;
    330        DLV68)  echo "range:3155:3156,range:4464:4465" ;;    # 
    331       DLV103)  echo "range:3155:3156,range:4543:4544" ;;    #
    332       DLV140)  echo "range:3155:3156,range:4606:4607" ;;    #
    333       DLV185)  echo "range:3155:3156,range:4799:4800" ;;    #
    334     esac
    335     # range:3154:3155  SST  Stainless Steel/IWSWater not a good choice for an envelope, just get BULK_ABSORB without going anywhere
    336 }




    op --idyb --gltf 3  

    op --drv3155 --gltf 3  

         hmm although this applies recursive select to a volume and contents, 
         instancing kicks in so other instanced volumes show up elsewhere 


/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GNodeLib/GTreePresent.txt::

     3142 [  5: 173/ 178]    0 ( 0)        __dd__Geometry__RPCSupport__TrivialComponents__lvNearDiagSquareIron0xc358910  near_diagonal_square_iron0xbf5f3f8   
     3143 [  5: 174/ 178]    0 ( 0)        __dd__Geometry__RPCSupport__TrivialComponents__lvNearDiagSquareIron0xc358910  near_diagonal_square_iron0xbf5f3f8   
     3144 [  5: 175/ 178]    0 ( 0)        __dd__Geometry__RPCSupport__TrivialComponents__lvNearDiagSquareIron0xc358910  near_diagonal_square_iron0xbf5f3f8   
     3145 [  5: 176/ 178]    0 ( 0)        __dd__Geometry__RPCSupport__TrivialComponents__lvNearDiagSquareIron0xc358910  near_diagonal_square_iron0xbf5f3f8   
     3146 [  5: 177/ 178]    0 ( 0)        __dd__Geometry__RPCSupport__TrivialComponents__lvNearDiagSquareIron0xc358910  near_diagonal_square_iron0xbf5f3f8   
     3147 [  2:   1/   2]   10 ( 0)     __dd__Geometry__Sites__lvNearHallBot0xbf89c60  near_hall_bot0xbf3d718   
     3148 [  3:   0/  10]    9 ( 0)      __dd__Geometry__Pool__lvNearPoolDead0xc2dc490  near_pool_dead_box0xbf8a280   
     3149 [  4:   0/   9]    9 ( 0)       __dd__Geometry__Pool__lvNearPoolLiner0xc21e9d0  near_pool_liner_box0xc2dcc28   
     3150 [  5:   0/   9] 2938 ( 0)        __dd__Geometry__Pool__lvNearPoolOWS0xbf93840  near_pool_ows_box0xbf8c8a8   
     3151 [  6:   0/2938]    9 ( 0)         __dd__Geometry__Pool__lvNearPoolCurtain0xc2ceef0  near_pool_curtain_box0xc2cef48   
     3152 [  7:   0/   9] 1619 ( 0)          __dd__Geometry__Pool__lvNearPoolIWS0xc28bc60  near_pool_iws_box0xc288ce8   
     3153 [  8:   0/1619]   11 ( 0)           __dd__Geometry__AD__lvADE0xc2a78c0  ade0xc2a7438   
     3154 [  9:   0/  11]    4 ( 0)            __dd__Geometry__AD__lvSST0xc234cd0  sst0xbf4b060   
     3155 [ 10:   0/   4]  520 ( 0)             __dd__Geometry__AD__lvOIL0xbf5e0b8  oil0xbf5ed48   
     3156 [ 11:   0/ 520]    3 ( 0)              __dd__Geometry__AD__lvOAV0xbf1c760  oav0xc2ed7c8   
     3157 [ 12:   0/   3]   35 ( 0)               __dd__Geometry__AD__lvLSO0xc403e40  lso0xc028a38   
     3158 [ 13:   0/  35]    2 ( 0)                __dd__Geometry__AD__lvIAV0xc404ee8  iav0xc346f90   
     3159 [ 14:   0/   2]    0 ( 0)                 __dd__Geometry__AD__lvGDS0xbf6cbb8  gds0xc28d3f0   
     3160 [ 14:   1/   2]    0 ( 0)                 __dd__Geometry__AdDetails__lvOcrGdsInIav0xbf6dd58  OcrGdsInIav0xc405b10   
     3161 [ 13:   1/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvIavTopHub0xc129d88  IavTopHub0xc405968   
     3162 [ 13:   2/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvCtrGdsOflBotClp0xc407eb0  CtrGdsOflBotClp0xbf5dec0   
     3163 [ 13:   3/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvCtrGdsOflTfbInLso0xbfa0728  CtrGdsOflTfbInLso0xbfa2d30   
     3164 [ 13:   4/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvCtrGdsOflInLso0xc28cc88  CtrGdsOflInLso0xbfa1178   
     3165 [ 13:   5/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvOcrGdsPrt0xc352630  OcrGdsPrt0xc352518   
     3166 [ 13:   6/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvCtrGdsOflBotClp0xc407eb0  CtrGdsOflBotClp0xbf5dec0   
     3167 [ 13:   7/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvOcrGdsTfbInLso0xc3529c0  OcrGdsTfbInLso0xbfa2370   







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


