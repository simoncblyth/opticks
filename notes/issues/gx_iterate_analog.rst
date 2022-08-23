gx_iterate_analog
====================

* prev :doc:`review_QEvent_SEvt_sevent_lifecycle_allocations_etc`

* three gx scripts : gxt.sh gxs.sh gxr.sh all run from the same saved geocache geometry. 
* analog iteration : plucking low hanging fruit to reduce runtime
* full JUNO total runtime now under 1s for all three. 

Iteration Procedure
---------------------

* run with logging enabled in several classes, eg:: 

    loglevels()
    {
        export Dummy=INFO
        #export SGeoConfig=INFO
        export SEventConfig=INFO
        export SEvt=INFO          # lots of AddGenstep output, messing with timings 
                                  #  actually little timing effect, just feels like it over network
        #export Ctx=INFO
        export QSim=INFO
        export QBase=INFO
        export SSim=INFO
        export SBT=INFO
        export IAS_Builder=INFO
        #export QEvent=INFO 
        export CSGOptiX=INFO
        export G4CXOpticks=INFO 
        export CSGFoundry=INFO
        #export GInstancer=INFO
        #export X4PhysicalVolume=INFO
        #export U4VolumeMaker=INFO
    }
    loglevels


* use analog to find where the time is going 
* change the logging positions to bracket calls that are taking the time  
* fix low handling fruit 

  1. CSGFoundry::inst_find_unique reduced from 20s to ~0s : it was finding uniques over all instances uneecessarily 
  2. IAS_Builder::CollectInstances using a map to cache the SBT offsets reduces time for all inst from 0.42s to ~0s  


All three now dominated by initial CUDA access time of ~0.32s (~0.25s with one GPU)
-------------------------------------------------------------------------------------

* CUDA first contact ~0.32s is dominant for all three 
* using CUDA_VISIBLE_DEVICES=0 OR 1 reduces that a little to 0.25s 
* this time is with nvidia-persistenced running (without it latency more like 1.5s) 


Results
---------

::

    epsilon:~ blyth$ gx
    /Users/blyth/opticks/g4cx
    epsilon:g4cx blyth$ ./gxt.sh grablog 
    epsilon:g4cx blyth$ ./gxt.sh analog

Log lines with delta time more than 2% of total time::

    epsilon:g4cx blyth$ ./gx.sh analog
    analog log G4CXSimulateTest.log
    repr(log[2])
                     timestamp :         DTS-prev :         DFS-frst :path:G4CXSimulateTest.log pc_cut:2 
    2022-08-23 23:46:28.045000 :                  :      0.0000[  0] : INFO  [57470] [main@31] [ cu first 
    2022-08-23 23:46:28.359000 :      0.3140[ 39] :      0.3140[ 39] : INFO  [57470] [main@33] ] cu first 
    2022-08-23 23:46:28.571000 :      0.1770[ 22] :      0.5260[ 65] : INFO  [57470] [QSim::UploadComponents@111] ] new QRng 
    2022-08-23 23:46:28.590000 :      0.0180[  2] :      0.5450[ 67] : INFO  [57470] [QSim::UploadComponents@128] QBnd src NP  dtype <f4(45, 4, 2, 761, 4, ) size 1095840 uifc f ebyte 4 shape.size 5 data.size 4383360 meta.size 69 names.size 45 tex QTex width 761 height 360 texObj 1 meta 0x3069a00 d_meta 0x7f3e9dc01000 tex 0x3069990
    2022-08-23 23:46:28.672000 :      0.0720[  9] :      0.6270[ 78] : INFO  [57470] [CSGOptiX::initCtx@322] ]
    2022-08-23 23:46:28.696000 :      0.0230[  3] :      0.6510[ 81] : INFO  [57470] [CSGOptiX::initPIP@333] ]
    2022-08-23 23:46:28.805000 :      0.0350[  4] :      0.7600[ 94] : INFO  [57470] [CSGFoundry::getFrame@2880]  fr sframe::desc inst 0 frs -1
    2022-08-23 23:46:28.839000 :      0.0200[  2] :      0.7940[ 98] : INFO  [57470] [CSGOptiX::launch@794]  (width, height, depth) ( 1920,1080,1) 0.0201
    2022-08-23 23:46:28.853000 :      0.0000[  0] :      0.8080[100] : INFO  [57470] [G4CXOpticks::saveEvent@422] ]
    === ./gxt.sh : analog log G4CXSimtraceTest.log
    repr(log[2])
                     timestamp :         DTS-prev :         DFS-frst :path:G4CXSimtraceTest.log pc_cut:2 
    2022-08-24 02:16:00.141000 :                  :      0.0000[  0] : INFO  [74325] [main@18] [ cu first 
    2022-08-24 02:16:00.444000 :      0.3030[ 31] :      0.3030[ 31] : INFO  [74325] [main@20] ] cu first 
    2022-08-24 02:16:00.663000 :      0.1780[ 18] :      0.5220[ 54] : INFO  [74325] [QSim::UploadComponents@111] ] new QRng 
    2022-08-24 02:16:00.765000 :      0.0730[  8] :      0.6240[ 65] : INFO  [74325] [CSGOptiX::initCtx@321]  ctx.desc Properties::desc
    2022-08-24 02:16:00.789000 :      0.0240[  2] :      0.6480[ 67] : INFO  [74325] [CSGOptiX::initPIP@333] ]
    2022-08-24 02:16:00.902000 :      0.0480[  5] :      0.7610[ 79] : INFO  [74325] [CSGFoundry::getFrame@2880]  fr sframe::desc inst 0 frs -1
    2022-08-24 02:16:01.072000 :      0.1130[ 12] :      0.9310[ 96] : INFO  [74325] [SEvt::gather@1378]  k        simtrace a  <f4(627000, 4, 4, )
    2022-08-24 02:16:01.105000 :      0.0320[  3] :      0.9640[100] : INFO  [74325] [SEvt::save@1505] ] fold.save 
    2022-08-24 02:16:01.106000 :      0.0000[  0] :      0.9650[100] : INFO  [74325] [G4CXOpticks::saveEvent@422] ]
    === ./gxr.sh : analog log G4CXRenderTest.log
    repr(log[2])
                     timestamp :         DTS-prev :         DFS-frst :path:G4CXRenderTest.log pc_cut:2 
    2022-08-24 02:16:12.185000 :                  :      0.0000[  0] : INFO  [74368] [main@22] [ cu first 
    2022-08-24 02:16:12.507000 :      0.3220[ 46] :      0.3220[ 46] : INFO  [74368] [main@24] ] cu first 
    2022-08-24 02:16:12.625000 :      0.0820[ 12] :      0.4400[ 63] : INFO  [74368] [CSGOptiX::initCtx@321]  ctx.desc Properties::desc
    2022-08-24 02:16:12.649000 :      0.0240[  3] :      0.4640[ 67] : INFO  [74368] [CSGOptiX::initPIP@333] ]
    2022-08-24 02:16:12.759000 :      0.0350[  5] :      0.5740[ 82] : INFO  [74368] [CSGFoundry::getFrame@2880]  fr sframe::desc inst 0 frs -1
    2022-08-24 02:16:12.880000 :      0.0970[ 14] :      0.6950[100] : INFO  [74368] [Frame::snap@155] ] writeJPG 
    2022-08-24 02:16:12.881000 :      0.0000[  0] :      0.6960[100] : INFO  [74368] [G4CXOpticks::render@397] ]
    epsilon:g4cx blyth$ 




CSGFoundry::inst_find_unique taking lots of time (21s) for little benefit
------------------------------------------------------------------------------

Avoid finding unique ins_index, sensor_identifier, sensor_index as those
are not used.  Shaving 21s::

    2022-08-22 03:03:32.535 INFO  [378584] [CSGFoundry::upload@2615] [ inst_find_unique 
    2022-08-22 03:03:32.539 INFO  [378584] [CSGFoundry::upload@2617] ] inst_find_unique 


::

    2022-08-22 02:29:24.822 INFO  [364740] [CSGOptiX::InitGeo@168] [
    2022-08-22 02:29:24.822 INFO  [364740] [CSGFoundry::upload@2610] [ inst_find_unique 
    2022-08-22 02:29:45.208 INFO  [364740] [CSGFoundry::upload@2612] ] inst_find_unique 
    2022-08-22 02:29:45.209 INFO  [364740] [CSGFoundry::upload@2613] CSGFoundry  num_total 10 num_solid 10 num_prim 3248 num_node 23518 num_plan 0 num_tran 8159 num_itra 8159 num_inst 48477 ins 48477 gas 10 sensor_identifier 45613 sensor_index 45613 meshname 139 mmlabel 10 mtime 1661012280 mtimestamp 20220821_001800 sim Y
    2022-08-22 02:29:45.209 INFO  [364740] [CSGFoundry::upload@2622] [ CU::UploadArray 
    2022-08-22 02:29:45.219 INFO  [364740] [CSGFoundry::upload@2627] ] CU::UploadArray 
    2022-08-22 02:29:45.219 INFO  [364740] [CSGFoundry::upload@2638] ]
    2022-08-22 02:29:45.219 INFO  [364740] [CSGOptiX::InitGeo@170] ]


