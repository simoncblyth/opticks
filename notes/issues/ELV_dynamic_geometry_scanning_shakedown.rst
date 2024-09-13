ELV_dynamic_geometry_scanning_shakedown
========================================



Find the ELV scanning machinery
---------------------------------

::

    P[blyth@localhost CSGOptiX]$ grep ELV *.sh 

    cxr_overview.sh:   vim -R /tmp/elv_txt.txt           ## view txt table to see the ELV exclusion names
    cxr_overview.sh:export ELV=${ELV:-$elv}
    cxr_overview.sh:## CAUTION: EMM(SBit) and ELV(SBitSet) lingos similar, but not the same. TODO: unify them  
    cxr_overview.sh:export NAMEPREFIX=cxr_overview_emm_${EMM}_elv_${ELV}_moi_      # MOI gets appended by the executable

    cxr_scan.sh:Repeats a script such as cxr_overview.sh with EMM or ELV
    cxr_scan.sh:        ELV=$e $DIR/$SCRIPT.sh $*

    cxr_view.sh:export ELV=${ELV:-$elv}
    cxr_view.sh:## CAUTION: EMM(SBit) and ELV(SBitSet) lingos similar, but not the same. TODO: unify them  
    cxr_view.sh:nameprefix=cxr_view_emm_${EMM}_elv_${ELV}
    cxr_view.sh:export TOPLINE="./cxr_view.sh $MOI      # EYE $EYE LOOK $LOOK UP $UP      EMM $EMM ELV $ELV  $stamp  $version " 

    elv.sh:elv.sh : analysis of ELV scan metadata
    P[blyth@localhost CSGOptiX]$ 



Analysis of ELV scan metadata
------------------------------

bin/BASE_grab.sh::

     32 if [ "${arg/jstab}" != "$arg" ]; then
     33 
     34     echo $BASH_SOURCE jstab     
     35     #jsons=($(ls -1t $(find $BASE -name '*.json')))
     36     #for json in ${jsons[*]} ; do echo $json ; done  
     37 
     38     globptn="$BASE/cxr_overview*elv*.jpg"
     39     refjpgpfx="/env/presentation/cxr/cxr_overview"
     40 
     41     ${IPYTHON:-ipython} --pdb  $OPTICKS_HOME/ana/snap.py --  --globptn "$globptn" --refjpgpfx "$refjpgpfx" $SNAP_ARGS
     42 fi
     43 




Test Scans on workstations P and A
------------------------------------

::

    P[blyth@localhost opticks]$ ~/o/CSGOptiX/cxr_scan.sh
    2024-09-11 17:14:04.165 INFO  [159875] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t0,_elv_t_moi__ALL.jpg :     0.0102 1:NVIDIA_TITAN_RTX 
    2024-09-11 17:14:07.637 INFO  [159980] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t1,_elv_t_moi__ALL.jpg :     0.0148 1:NVIDIA_TITAN_RTX 
    2024-09-11 17:14:11.157 INFO  [160105] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t2,_elv_t_moi__ALL.jpg :     0.0158 1:NVIDIA_TITAN_RTX 
    2024-09-11 17:14:14.632 INFO  [160210] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t3,_elv_t_moi__ALL.jpg :     0.0115 1:NVIDIA_TITAN_RTX 
    2024-09-11 17:14:18.201 INFO  [160338] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t4,_elv_t_moi__ALL.jpg :     0.0115 1:NVIDIA_TITAN_RTX 
    2024-09-11 17:14:21.728 INFO  [160443] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t5,_elv_t_moi__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-11 17:14:25.280 INFO  [160570] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t6,_elv_t_moi__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-11 17:14:28.751 INFO  [160676] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t7,_elv_t_moi__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-11 17:14:32.366 INFO  [160800] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t8,_elv_t_moi__ALL.jpg :     0.0145 1:NVIDIA_TITAN_RTX 
    2024-09-11 17:14:35.816 INFO  [160916] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t9,_elv_t_moi__ALL.jpg :     0.0121 1:NVIDIA_TITAN_RTX 
    2024-09-11 17:14:39.217 INFO  [161031] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t10,_elv_t_moi__ALL.jpg :     0.0119 1:NVIDIA_TITAN_RTX 
    P[blyth@localhost opticks]$ 

    P[blyth@localhost opticks]$ ~/o/CSGOptiX/cxr_scan.sh
    2024-09-11 17:18:01.625 INFO  [168121] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t0,_elv_t_moi__ALL00000.jpg :     0.0091 1:NVIDIA_TITAN_RTX 
    2024-09-11 17:18:05.177 INFO  [168248] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t1,_elv_t_moi__ALL00000.jpg :     0.0134 1:NVIDIA_TITAN_RTX 
    2024-09-11 17:18:08.649 INFO  [168351] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t2,_elv_t_moi__ALL00000.jpg :     0.0115 1:NVIDIA_TITAN_RTX 
    2024-09-11 17:18:12.065 INFO  [168480] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t3,_elv_t_moi__ALL00000.jpg :     0.0112 1:NVIDIA_TITAN_RTX 
    2024-09-11 17:18:15.659 INFO  [168584] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t4,_elv_t_moi__ALL00000.jpg :     0.0118 1:NVIDIA_TITAN_RTX 
    2024-09-11 17:18:19.182 INFO  [168711] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t5,_elv_t_moi__ALL00000.jpg :     0.0121 1:NVIDIA_TITAN_RTX 
    2024-09-11 17:18:22.681 INFO  [168814] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t6,_elv_t_moi__ALL00000.jpg :     0.0114 1:NVIDIA_TITAN_RTX 
    2024-09-11 17:18:26.394 INFO  [168943] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t7,_elv_t_moi__ALL00000.jpg :     0.0116 1:NVIDIA_TITAN_RTX 
    2024-09-11 17:18:29.985 INFO  [169071] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t8,_elv_t_moi__ALL00000.jpg :     0.0140 1:NVIDIA_TITAN_RTX 
    2024-09-11 17:18:33.565 INFO  [169174] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t9,_elv_t_moi__ALL00000.jpg :     0.0115 1:NVIDIA_TITAN_RTX 
    2024-09-11 17:18:36.962 INFO  [169301] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t10,_elv_t_moi__ALL00000.jpg :     0.0115 1:NVIDIA_TITAN_RTX 
    P[blyth@localhost opticks]$ 


Reruns to check output path fixes::

    P[blyth@localhost CSGOptiX]$ ~/o/CSGOptiX/cxr_scan.sh
    2024-09-11 20:07:23.195 INFO  [5692] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t0,_elv_t_moi__ALL.jpg :     0.0092 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:07:26.801 INFO  [5795] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t1,_elv_t_moi__ALL.jpg :     0.0133 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:07:30.393 INFO  [5926] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t2,_elv_t_moi__ALL.jpg :     0.0122 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:07:34.018 INFO  [6053] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t3,_elv_t_moi__ALL.jpg :     0.0111 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:07:37.520 INFO  [6161] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t4,_elv_t_moi__ALL.jpg :     0.0108 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:07:40.978 INFO  [6302] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t5,_elv_t_moi__ALL.jpg :     0.0139 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:07:44.469 INFO  [6406] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t6,_elv_t_moi__ALL.jpg :     0.0115 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:07:47.991 INFO  [6547] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t7,_elv_t_moi__ALL.jpg :     0.0117 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:07:51.631 INFO  [6659] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t8,_elv_t_moi__ALL.jpg :     0.0140 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:07:55.283 INFO  [6792] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t9,_elv_t_moi__ALL.jpg :     0.0117 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:07:58.921 INFO  [6901] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t10,_elv_t_moi__ALL.jpg :     0.0115 1:NVIDIA_TITAN_RTX 

    P[blyth@localhost CSGOptiX]$ ~/o/CSGOptiX/cxr_scan.sh
    2024-09-11 20:09:08.303 INFO  [8808] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t0,_elv_t_moi__ALL00000.jpg :     0.0093 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:09:11.955 INFO  [8925] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t1,_elv_t_moi__ALL00000.jpg :     0.0148 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:09:15.458 INFO  [9041] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t2,_elv_t_moi__ALL00000.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:09:18.980 INFO  [9166] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t3,_elv_t_moi__ALL00000.jpg :     0.0113 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:09:22.509 INFO  [9272] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t4,_elv_t_moi__ALL00000.jpg :     0.0110 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:09:26.040 INFO  [9399] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t5,_elv_t_moi__ALL00000.jpg :     0.0116 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:09:29.731 INFO  [9503] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t6,_elv_t_moi__ALL00000.jpg :     0.0115 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:09:33.461 INFO  [9630] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t7,_elv_t_moi__ALL00000.jpg :     0.0116 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:09:37.001 INFO  [9757] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t8,_elv_t_moi__ALL00000.jpg :     0.0139 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:09:40.843 INFO  [9860] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t9,_elv_t_moi__ALL00000.jpg :     0.0125 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:09:44.327 INFO  [9992] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t10,_elv_t_moi__ALL00000.jpg :     0.0114 1:NVIDIA_TITAN_RTX 
    P[blyth@localhost CSGOptiX]$ 

    P[blyth@localhost CSGOptiX]$ ~/o/CSGOptiX/cxr_scan.sh
    2024-09-11 20:12:13.470 INFO  [14100] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t0,_elv_t_moi__ALL00001.jpg :     0.0092 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:12:17.105 INFO  [14230] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t1,_elv_t_moi__ALL00001.jpg :     0.0134 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:12:20.614 INFO  [14344] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t2,_elv_t_moi__ALL00001.jpg :     0.0115 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:12:24.085 INFO  [14481] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t3,_elv_t_moi__ALL00001.jpg :     0.0112 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:12:27.707 INFO  [14616] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t4,_elv_t_moi__ALL00001.jpg :     0.0108 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:12:31.284 INFO  [14754] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t5,_elv_t_moi__ALL00001.jpg :     0.0115 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:12:35.331 INFO  [14890] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t6,_elv_t_moi__ALL00001.jpg :     0.0114 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:12:38.947 INFO  [15009] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t7,_elv_t_moi__ALL00001.jpg :     0.0116 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:12:42.784 INFO  [15130] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t8,_elv_t_moi__ALL00001.jpg :     0.0191 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:12:46.354 INFO  [15262] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t9,_elv_t_moi__ALL00001.jpg :     0.0184 1:NVIDIA_TITAN_RTX 
    2024-09-11 20:12:50.077 INFO  [15390] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t10,_elv_t_moi__ALL00001.jpg :     0.0113 1:NVIDIA_TITAN_RTX 
    P[blyth@localhost CSGOptiX]$ 



Rerun with different GPU and Release build::

    A[blyth@localhost opticks]$ ~/o/CSGOptiX/cxr_scan.sh 
    2024-09-11 17:37:41.521 INFO  [410909] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/ALL/scan-emm/cxr_overview_emm_t0,_elv_t_moi__ALL.jpg :     0.0020 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-11 17:37:43.851 INFO  [410932] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/ALL/scan-emm/cxr_overview_emm_t1,_elv_t_moi__ALL.jpg :     0.0030 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-11 17:37:45.911 INFO  [410955] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/ALL/scan-emm/cxr_overview_emm_t2,_elv_t_moi__ALL.jpg :     0.0030 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-11 17:37:48.009 INFO  [410978] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/ALL/scan-emm/cxr_overview_emm_t3,_elv_t_moi__ALL.jpg :     0.0030 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-11 17:37:50.112 INFO  [411001] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/ALL/scan-emm/cxr_overview_emm_t4,_elv_t_moi__ALL.jpg :     0.0029 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-11 17:37:52.205 INFO  [411024] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/ALL/scan-emm/cxr_overview_emm_t5,_elv_t_moi__ALL.jpg :     0.0030 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-11 17:37:54.323 INFO  [411047] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/ALL/scan-emm/cxr_overview_emm_t6,_elv_t_moi__ALL.jpg :     0.0031 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-11 17:37:56.444 INFO  [411070] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/ALL/scan-emm/cxr_overview_emm_t7,_elv_t_moi__ALL.jpg :     0.0035 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-11 17:37:58.547 INFO  [411093] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/ALL/scan-emm/cxr_overview_emm_t8,_elv_t_moi__ALL.jpg :     0.0035 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-11 17:38:00.627 INFO  [411116] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/ALL/scan-emm/cxr_overview_emm_t9,_elv_t_moi__ALL.jpg :     0.0040 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-11 17:38:02.738 INFO  [411139] [CSGOptiX::render_save_@1271] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/ALL/scan-emm/cxr_overview_emm_t10,_elv_t_moi__ALL.jpg :     0.0030 0:NVIDIA_RTX_5000_Ada_Generation 
    A[blyth@localhost opticks]$ 




FIXED : naming inconsistency in output
-----------------------------------------------------

::

    P[blyth@localhost scan-emm]$ l
    total 8924
    356 -rw-rw-r--. 1 blyth blyth 361617 Sep 11 17:18 cxr_overview_emm_t10,_elv_t_moi__ALL00000.jpg
      4 -rw-rw-r--. 1 blyth blyth   1241 Sep 11 17:18 cxr_overview_emm_t10,_elv_t_moi__ALL00000.json
     12 -rw-rw-r--. 1 blyth blyth   9955 Sep 11 17:18 cxr_overview_emm_t10,_elv_t_moi__ALL.log
      4 -rw-rw-r--. 1 blyth blyth     32 Sep 11 17:18 cxr_overview_emm_t10,_elv_t_moi__ALL_meta.txt
      4 -rw-rw-r--. 1 blyth blyth    512 Sep 11 17:18 cxr_overview_emm_t10,_elv_t_moi__ALL.npy
     12 drwxr-xr-x. 2 blyth blyth   8192 Sep 11 17:18 .



FIXED : on A no CUDA capable device ? fix by removing the CVD at script level
-------------------------------------------------------------------------------

::

    A[blyth@localhost opticks]$ ~/o/CSGOptiX/cxr_overview.sh 
                   stamp : 2024-09-11 17:30 
                 version : 80000 
                 TOPLINE : ./cxr_overview.sh    # EYE -0.6,0,0,1 MOI ALL ZOOM 1.5 stamp 2024-09-11 17:30 version 80000 done 
                 BOTLINE :  GEOM J_2024aug27 RELDIR cam_0_tmin_0.4 NAMEPREFIX cxr_overview_emm_t0_elv_t_moi_ SCAN   
           CVD :  
    CUDA_VISIBLE_DEVICES : 1 
           EMM : t0 
           MOI : ALL 
           EYE : -0.6,0,0,1 
           TOP : i0 
           SLA :  
           CAM : perspective 
          TMIN : 0.4 
          ZOOM : 1.5 
    CAMERATYPE : 0 
    OPTICKS_GEOM : cxr_overview 
    OPTICKS_RELDIR : cam_0_tmin_0.4 
          SIZE : 1280,720,1 
     SIZESCALE : 1.5 
        CFBASE :  
    OPTICKS_OUT_FOLD : /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/80000 
    OPTICKS_OUT_NAME : ALL 
    /data1/blyth/local/opticks_Release/lib/CSGOptiXRenderTest
    /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest
    ==== cxr.sh render : CSGOptiXRenderTest
    terminate called after throwing an instance of 'CUDA_Exception'
      what():  CUDA call (cudaMalloc(reinterpret_cast<void**>( &d_array ), num_items*sizeof(T) ) ) failed with error: 'no CUDA-capable device is detected' (/home/blyth/opticks/CSG/CU.cc:56)

    cxr.sh: line 177: 410635 Aborted                 (core dumped) CSGOptiXRenderTest
    ==== cxr.sh render : CSGOptiXRenderTest : rc 134
    A[blyth@localhost opticks]$ 
    A[blyth@localhost opticks]$ 




ana shakedown : Fixed issue 1 : descriptions are all "ALL" 
--------------------------------------------------------------

::

    P[blyth@localhost CSGOptiX]$ ~/o/CSGOptiX/elv.sh txt
                    BASE : /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm 
    /data/blyth/junotop/opticks/CSGOptiX/../bin/BASE_grab.sh jstab
    [2024-09-11 21:39:13,514] p153898 {/data/blyth/junotop/opticks/ana/snap.py:469} INFO - globptn /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview*elv*.jpg 
    [2024-09-11 21:39:13,515] p153898 {/data/blyth/junotop/opticks/ana/snap.py:312} INFO - cfptn $HOME/.opticks/GEOM/$GEOM/CSGFoundry cfdir /home/blyth/.opticks/GEOM/J_2024aug27/CSGFoundry 
    [2024-09-11 21:39:13,515] p153898 {/data/blyth/junotop/opticks/ana/snap.py:315} INFO - mmlabel_path /home/blyth/.opticks/GEOM/J_2024aug27/CSGFoundry/mmlabel.txt 
    [2024-09-11 21:39:13,515] p153898 {/data/blyth/junotop/opticks/ana/snap.py:319} INFO - meshname_path /home/blyth/.opticks/GEOM/J_2024aug27/CSGFoundry/meshname.txt 
    [2024-09-11 21:39:13,515] p153898 {/data/blyth/junotop/opticks/ana/snap.py:252} INFO - globptn /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview*elv*.jpg 
    [2024-09-11 21:39:13,516] p153898 {/data/blyth/junotop/opticks/ana/snap.py:254} INFO - globptn raw_paths 33 : 1st /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t0,_elv_t_moi__ALL.jpg 
    [2024-09-11 21:39:13,516] p153898 {/data/blyth/junotop/opticks/ana/snap.py:256} INFO - after is_valid filter len(paths): 33 
    [2024-09-11 21:39:13,518] p153898 {/data/blyth/junotop/opticks/ana/snap.py:361} INFO - all_snaps:33 
    [2024-09-11 21:39:13,518] p153898 {/data/blyth/junotop/opticks/ana/snap.py:300} INFO - all_snaps 33 selectspec all snaps 33 SNAP_LIMIT 512 lim_snaps 33 
    [2024-09-11 21:39:13,518] p153898 {/data/blyth/junotop/opticks/ana/snap.py:375} INFO - after selectmode:elv selectspec:all snaps:33 
    [2024-09-11 21:39:13,518] p153898 {/data/blyth/junotop/opticks/ana/snap.py:497} INFO - --out writing to /tmp/elv_txt.txt 
    /tmp/elv_txt.txt
    idx         -e        time(s)           relative         enabled geometry description                                              
      0          t         0.0092             0.4788         ALL                                                                       
      1          t         0.0092             0.4807         ALL                                                                       
      2          t         0.0093             0.4884         ALL                                                                       
      3          t         0.0108             0.5657         ALL                                                                       
      4          t         0.0108             0.5659         ALL                                                                       
      5          t         0.0110             0.5758         ALL                                                                       
      6          t         0.0111             0.5827         ALL                                                                       
      7          t         0.0112             0.5842         ALL                                                                       
      8          t         0.0113             0.5905         ALL                                                                       
      9          t         0.0113             0.5936         ALL                                                                       
     10          t         0.0114             0.5965         ALL           




Issue 2 : PILOT ERROR : EMM mode relative column not working
------------------------------------------------------------------

Not an issue. Need envvar to specify the CANDLE and need to have previously 
run the candle render to have the metadata to act as the candle. 

::

    ~/o/CSGOptiX/elv.sh txt

    CANDLE=t0 ~/o/CSGOptiX/elv.sh txt
    CANDLE=1,2,3,4 ~/o/CSGOptiX/elv.sh txt


::

    P[blyth@localhost opticks]$ CANDLE=t0 ~/o/CSGOptiX/elv.sh txt

                    BASE : /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm 
    /data/blyth/junotop/opticks/CSGOptiX/../bin/BASE_grab.sh jstab
    [2024-09-12 17:10:04,284] p46080 {/data/blyth/junotop/opticks/ana/snap.py:492} INFO - globptn /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview*elv*.jpg 
    [2024-09-12 17:10:04,284] p46080 {/data/blyth/junotop/opticks/ana/snap.py:325} INFO - cfptn $HOME/.opticks/GEOM/$GEOM/CSGFoundry cfdir /home/blyth/.opticks/GEOM/J_2024aug27/CSGFoundry 
    [2024-09-12 17:10:04,284] p46080 {/data/blyth/junotop/opticks/ana/snap.py:328} INFO - mmlabel_path /home/blyth/.opticks/GEOM/J_2024aug27/CSGFoundry/mmlabel.txt 
    [2024-09-12 17:10:04,284] p46080 {/data/blyth/junotop/opticks/ana/snap.py:332} INFO - meshname_path /home/blyth/.opticks/GEOM/J_2024aug27/CSGFoundry/meshname.txt 
    [2024-09-12 17:10:04,284] p46080 {/data/blyth/junotop/opticks/ana/snap.py:265} INFO - globptn /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview*elv*.jpg 
    [2024-09-12 17:10:04,285] p46080 {/data/blyth/junotop/opticks/ana/snap.py:267} INFO - globptn raw_paths 46 : 1st /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t0,_elv_t_moi__ALL.jpg 
    [2024-09-12 17:10:04,285] p46080 {/data/blyth/junotop/opticks/ana/snap.py:269} INFO - after is_valid filter len(paths): 46 
    [2024-09-12 17:10:04,288] p46080 {/data/blyth/junotop/opticks/ana/snap.py:378} INFO - all_snaps:46 candle:t0 n_candle:1 selectmode:emm 
    [2024-09-12 17:10:04,288] p46080 {/data/blyth/junotop/opticks/ana/snap.py:392} INFO - after selectmode:emm selectspec:all snaps:46 
    [2024-09-12 17:10:04,288] p46080 {/data/blyth/junotop/opticks/ana/snap.py:520} INFO - --out writing to /tmp/emm_txt.txt 
    /tmp/emm_txt.txt
    idx         -e        time(s)           relative         enabled geometry description                                              
      0         5,         0.0015             0.1298         ONLY: 1:sStrutBallhead                                                    
      1         9,         0.0016             0.1389         ONLY: 130:sPanel                                                          
      2         7,         0.0017             0.1441         ONLY: 1:base_steel                                                        
      3         6,         0.0017             0.1445         ONLY: 1:uni1                                                              
      4        10,         0.0017             0.1491         ONLY: 322:solidSJCLSanchor                                                
      5         8,         0.0018             0.1544         ONLY: 1:uni_acrylic1                                                      
      6         4,         0.0025             0.2145         ONLY: 4:mask_PMT_20inch_vetosMask_virtual                                 
      7         3,         0.0055             0.4736         ONLY: 12:HamamatsuR12860sMask_virtual                                     
      8         2,         0.0061             0.5242         ONLY: 9:NNVTMCPPMTsMask_virtual                                           
      9         1,         0.0064             0.5459         ONLY: 5:PMT_3inch_pmt_solid                                               
     10         0,         0.0086             0.7420         ONLY: 2896:sWorld                                                         
     11    1,2,3,4         0.0091             0.7840         ONLY PMT                                                                  
     12        t0,         0.0092             0.7864         EXCL: 2896:sWorld                                                         
     13        t0,         0.0092             0.7895         EXCL: 2896:sWorld                                                         
     14        t0,         0.0093             0.8021         EXCL: 2896:sWorld                                                         
     15        t4,         0.0108             0.9291         EXCL: 4:mask_PMT_20inch_vetosMask_virtual                                 
     16        t4,         0.0108             0.9295         EXCL: 4:mask_PMT_20inch_vetosMask_virtual                                 
     17        t4,         0.0110             0.9456         EXCL: 4:mask_PMT_20inch_vetosMask_virtual                                 
     18        t3,         0.0111             0.9571         EXCL: 12:HamamatsuR12860sMask_virtual                                     
     19        t3,         0.0112             0.9595         EXCL: 12:HamamatsuR12860sMask_virtual                                     
     20        t3,         0.0113             0.9698         EXCL: 12:HamamatsuR12860sMask_virtual                                     
     21       t10,         0.0113             0.9749         EXCL: 322:solidSJCLSanchor                                                
     22        t6,         0.0114             0.9797         EXCL: 1:uni1                                                              
     23       t10,         0.0114             0.9815         EXCL: 322:solidSJCLSanchor                                                
     24        t6,         0.0115             0.9863         EXCL: 1:uni1                                                              
     25        t2,         0.0115             0.9871         EXCL: 9:NNVTMCPPMTsMask_virtual                                           
     26       t10,         0.0115             0.9886         EXCL: 322:solidSJCLSanchor                                                
     27        t6,         0.0115             0.9886         EXCL: 1:uni1                                                              
     28        t5,         0.0115             0.9895         EXCL: 1:sStrutBallhead                                                    
     29        t7,         0.0116             0.9933         EXCL: 1:base_steel                                                        
     30        t7,         0.0116             0.9971         EXCL: 1:base_steel                                                        
     31        t5,         0.0116             0.9991         EXCL: 1:sStrutBallhead                                                    
     32         t0         0.0116             1.0000         ALL                                                                       
     33        t9,         0.0117             1.0027         EXCL: 130:sPanel                                                          
     34        t7,         0.0117             1.0042         EXCL: 1:base_steel                                                        
     35        t2,         0.0122             1.0462         EXCL: 9:NNVTMCPPMTsMask_virtual                                           
     36        t9,         0.0125             1.0720         EXCL: 130:sPanel                                                          
     37        t2,         0.0128             1.1020         EXCL: 9:NNVTMCPPMTsMask_virtual                                           
     38        t1,         0.0133             1.1461         EXCL: 5:PMT_3inch_pmt_solid                                               
     39        t1,         0.0134             1.1468         EXCL: 5:PMT_3inch_pmt_solid                                               
     40        t5,         0.0139             1.1901         EXCL: 1:sStrutBallhead                                                    
     41        t8,         0.0139             1.1930         EXCL: 1:uni_acrylic1                                                      
     42        t8,         0.0140             1.2007         EXCL: 1:uni_acrylic1                                                      
     43        t1,         0.0148             1.2754         EXCL: 5:PMT_3inch_pmt_solid                                               
     44        t9,         0.0184             1.5776         EXCL: 130:sPanel                                                          
     45        t8,         0.0191             1.6423         EXCL: 1:uni_acrylic1                                                      
    idx         -e        time(s)           relative         enabled geometry description                                              P[blyth@localhost opticks]$ 
    P[blyth@localhost opticks]$ 




    A[blyth@localhost bin]$ CANDLE=t0 ~/o/CSGOptiX/elv.sh txt
                    BASE : /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/ALL/scan-emm 
    /home/blyth/opticks/CSGOptiX/../bin/BASE_grab.sh jstab
    /home/blyth/opticks/CSGOptiX/../bin/BASE_grab.sh - IPYTHON NOT AVAILABLE - TRY PYTHON
    [2024-09-12 20:11:00,975] p18646 {/home/blyth/opticks/bin/../ana/snap.py:492} INFO - globptn /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/ALL/scan-emm/cxr_overview*elv*.jpg 
    [2024-09-12 20:11:00,975] p18646 {/home/blyth/opticks/bin/../ana/snap.py:325} INFO - cfptn $HOME/.opticks/GEOM/$GEOM/CSGFoundry cfdir /home/blyth/.opticks/GEOM/J_2024aug27/CSGFoundry 
    [2024-09-12 20:11:00,975] p18646 {/home/blyth/opticks/bin/../ana/snap.py:328} INFO - mmlabel_path /home/blyth/.opticks/GEOM/J_2024aug27/CSGFoundry/mmlabel.txt 
    [2024-09-12 20:11:00,976] p18646 {/home/blyth/opticks/bin/../ana/snap.py:332} INFO - meshname_path /home/blyth/.opticks/GEOM/J_2024aug27/CSGFoundry/meshname.txt 
    [2024-09-12 20:11:00,976] p18646 {/home/blyth/opticks/bin/../ana/snap.py:265} INFO - globptn /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/ALL/scan-emm/cxr_overview*elv*.jpg 
    [2024-09-12 20:11:00,977] p18646 {/home/blyth/opticks/bin/../ana/snap.py:267} INFO - globptn raw_paths 35 : 1st /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/ALL/scan-emm/cxr_overview_emm_t0,_elv_t_moi__ALL.jpg 
    [2024-09-12 20:11:00,977] p18646 {/home/blyth/opticks/bin/../ana/snap.py:269} INFO - after is_valid filter len(paths): 35 
    [2024-09-12 20:11:00,977] p18646 {/home/blyth/opticks/bin/../ana/snap.py:378} INFO - all_snaps:35 candle:t0 n_candle:1 selectmode:emm 
    [2024-09-12 20:11:00,977] p18646 {/home/blyth/opticks/bin/../ana/snap.py:392} INFO - after selectmode:emm selectspec:all snaps:35 
    [2024-09-12 20:11:00,978] p18646 {/home/blyth/opticks/bin/../ana/snap.py:520} INFO - --out writing to /tmp/emm_txt.txt 
    >>> 
    /tmp/emm_txt.txt
    idx         -e        time(s)           relative         enabled geometry description                                              
      0         5,         0.0001             0.0476         ONLY: 1:sStrutBallhead                                                    
      1         9,         0.0002             0.0559         ONLY: 130:sPanel                                                          
      2         7,         0.0002             0.0597         ONLY: 1:base_steel                                                        
      3        10,         0.0002             0.0645         ONLY: 322:solidSJCLSanchor                                                
      4         6,         0.0002             0.0685         ONLY: 1:uni1                                                              
      5         8,         0.0002             0.0690         ONLY: 1:uni_acrylic1                                                      
      6         1,         0.0003             0.1011         ONLY: 5:PMT_3inch_pmt_solid                                               
      7         4,         0.0004             0.1323         ONLY: 4:mask_PMT_20inch_vetosMask_virtual                                 
      8         3,         0.0012             0.3880         ONLY: 12:HamamatsuR12860sMask_virtual                                     
      9         2,         0.0015             0.4699         ONLY: 9:NNVTMCPPMTsMask_virtual                                           
     10        t0,         0.0020             0.6427         EXCL: 2896:sWorld                                                         
     11        t0,         0.0020             0.6527         EXCL: 2896:sWorld                                                         
     12    1,2,3,4         0.0021             0.6638         ONLY PMT                                                                  
     13         0,         0.0021             0.6740         ONLY: 2896:sWorld                                                         
     14        t4,         0.0029             0.9425         EXCL: 4:mask_PMT_20inch_vetosMask_virtual                                 
     15        t3,         0.0030             0.9490         EXCL: 12:HamamatsuR12860sMask_virtual                                     
     16        t4,         0.0030             0.9506         EXCL: 4:mask_PMT_20inch_vetosMask_virtual                                 
     17        t2,         0.0030             0.9569         EXCL: 9:NNVTMCPPMTsMask_virtual                                           
     18        t3,         0.0030             0.9676         EXCL: 12:HamamatsuR12860sMask_virtual                                     
     19       t10,         0.0030             0.9710         EXCL: 322:solidSJCLSanchor                                                
     20        t1,         0.0030             0.9712         EXCL: 5:PMT_3inch_pmt_solid                                               
     21       t10,         0.0030             0.9765         EXCL: 322:solidSJCLSanchor                                                
     22        t5,         0.0030             0.9775         EXCL: 1:sStrutBallhead                                                    
     23        t6,         0.0031             0.9817         EXCL: 1:uni1                                                              
     24        t2,         0.0031             0.9859         EXCL: 9:NNVTMCPPMTsMask_virtual                                           
     25        t1,         0.0031             0.9871         EXCL: 5:PMT_3inch_pmt_solid                                               
     26        t5,         0.0031             0.9911         EXCL: 1:sStrutBallhead                                                    
     27        t6,         0.0031             0.9953         EXCL: 1:uni1                                                              
     28         t0         0.0031             1.0000         ALL                                                                       




ELV scanning with cxr_overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cxr_overview viewpoint of the ELV ONLY is very variable for the torii, maybe because its
global and the axis aligned bbox changes size a lot due to different orientations
of the torii section. 

For the exclude one renders, they mostly all look the same with the distant cxr_overview. 

Presumably using the fixed chimmey viewpoint of cxr_view makes more sense for ELV scanning.


::

    P[blyth@localhost opticks]$ CANDLE=t ~/o/CSGOptiX/elv.sh txt
                    BASE : /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-elv 
    /data/blyth/junotop/opticks/CSGOptiX/../bin/BASE_grab.sh jstab
    [2024-09-12 20:57:34,978] p82566 {/data/blyth/junotop/opticks/ana/snap.py:492} INFO - globptn /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-elv/cxr_overview*elv*.jpg 
    [2024-09-12 20:57:34,978] p82566 {/data/blyth/junotop/opticks/ana/snap.py:325} INFO - cfptn $HOME/.opticks/GEOM/$GEOM/CSGFoundry cfdir /home/blyth/.opticks/GEOM/J_2024aug27/CSGFoundry 
    [2024-09-12 20:57:34,978] p82566 {/data/blyth/junotop/opticks/ana/snap.py:328} INFO - mmlabel_path /home/blyth/.opticks/GEOM/J_2024aug27/CSGFoundry/mmlabel.txt 
    [2024-09-12 20:57:34,978] p82566 {/data/blyth/junotop/opticks/ana/snap.py:332} INFO - meshname_path /home/blyth/.opticks/GEOM/J_2024aug27/CSGFoundry/meshname.txt 
    [2024-09-12 20:57:34,978] p82566 {/data/blyth/junotop/opticks/ana/snap.py:265} INFO - globptn /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-elv/cxr_overview*elv*.jpg 
    [2024-09-12 20:57:34,982] p82566 {/data/blyth/junotop/opticks/ana/snap.py:267} INFO - globptn raw_paths 303 : 1st /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-elv/cxr_overview_emm_t0_elv_0_moi__ALL.jpg 
    [2024-09-12 20:57:34,982] p82566 {/data/blyth/junotop/opticks/ana/snap.py:269} INFO - after is_valid filter len(paths): 303 
    [2024-09-12 20:57:34,998] p82566 {/data/blyth/junotop/opticks/ana/snap.py:378} INFO - all_snaps:303 candle:t n_candle:1 selectmode:elv 
    [2024-09-12 20:57:34,999] p82566 {/data/blyth/junotop/opticks/ana/snap.py:313} INFO - all_snaps 303 selectspec all snaps 303 SNAP_LIMIT 512 lim_snaps 303 
    [2024-09-12 20:57:34,999] p82566 {/data/blyth/junotop/opticks/ana/snap.py:392} INFO - after selectmode:elv selectspec:all snaps:303 
    [2024-09-12 20:57:35,001] p82566 {/data/blyth/junotop/opticks/ana/snap.py:520} INFO - --out writing to /tmp/elv_txt.txt 
    /tmp/elv_txt.txt
    idx         -e        time(s)           relative         enabled geometry description                                              
      0        113         0.0016             0.1366         ONLY: HamamatsuR12860_PMT_20inch_grid_solid                               
      1         51         0.0016             0.1373         ONLY: GLb2.equ_FlangeI_Web_FlangeII                                       
      2        297         0.0016             0.1417         ONLY: mask_PMT_20inch_vetosMask_virtual                                   
      3        100         0.0017             0.1426         ONLY: sStrutBallhead                                                      
      4         26         0.0017             0.1432         ONLY: GLw1.up02_up03_FlangeI_Web_FlangeII                                 
      5        130         0.0017             0.1459         ONLY: PMT_3inch_cntr_solid                                                
      6         25         0.0017             0.1480         ONLY: GLw1.up03_up04_FlangeI_Web_FlangeII                                 
      7        111         0.0017             0.1501         ONLY: HamamatsuR12860_PMT_20inch_inner_ring_solid                         
      8         75         0.0017             0.1504         ONLY: ZC2.A02_B02_FlangeI_Web_FlangeII                                    
      9        123         0.0017             0.1507         ONLY: NNVTMCPPMT_PMT_20inch_mcp_solid                                     
     10         33         0.0018             0.1510         ONLY: GLw2.bt04_bt05_FlangeI_Web_FlangeII                                 
     11        110         0.0018             0.1514         ONLY: HamamatsuR12860_PMT_20inch_inner_edge_solid                         
     12        127         0.0018             0.1544         ONLY: PMT_3inch_inner1_solid_ell_helper                                   
     13        295         0.0018             0.1545         ONLY: PMT_20inch_veto_inner_solid_1_2                                     
     14        108         0.0018             0.1551         ONLY: HamamatsuR12860_PMT_20inch_plate_solid                              
     15        109         0.0018             0.1563         ONLY: HamamatsuR12860_PMT_20inch_outer_edge_solid                         
     16         71         0.0018             0.1589         ONLY: GZ1.B03_04_FlangeI_Web_FlangeII                                     
     17        112         0.0019             0.1597         ONLY: HamamatsuR12860_PMT_20inch_dynode_tube_solid                        
     18        114         0.0019             0.1604         ONLY: HamamatsuR12860_PMT_20inch_shield_solid                             
     19         64         0.0019             0.1612         ONLY: GZ1.A02_03_FlangeI_Web_FlangeII                                     
     20         79         0.0019             0.1616         ONLY: ZC2.A06_B06_FlangeI_Web_FlangeII                                    
     21          3         0.0019             0.1617         ONLY: PoolCoversub                                                        
     22         43         0.0019             0.1618         ONLY: GLb2.up08_FlangeI_Web_FlangeII                                      
     23         84         0.0019             0.1623         ONLY: ZC2.B01_B01_FlangeI_Web_FlangeII                                    
     24         78         0.0019             0.1628         ONLY: ZC2.A05_B05_FlangeI_Web_FlangeII                                    
     25         90         0.0019             0.1637         ONLY: sTyvek_shell                                                        
     26         52         0.0019             0.1641         ONLY: GLb2.bt01_FlangeI_Web_FlangeII                                      
     27         91         0.0019             0.1650         ONLY: sDeadWater_shell                                                    
     28         31         0.0019             0.1653         ONLY: GLw3.bt02_bt03_FlangeI_Web_FlangeII                                 
     29        252         0.0019             0.1661         ONLY: svacSurftube_19V1_1                                                 
     30         72         0.0019             0.1671         ONLY: GZ1.B04_05_FlangeI_Web_FlangeII                                     
     31         55         0.0019             0.1673         ONLY: GLb2.bt04_FlangeI_Web_FlangeII                                      
     32        122         0.0019             0.1678         ONLY: NNVTMCPPMT_PMT_20inch_tube_solid                                    
     33         94         0.0019             0.1679         ONLY: solidSJReceiver                                                     
     34         45         0.0019             0.1679         ONLY: GLb2.up06_FlangeI_Web_FlangeII                                      
     35        121         0.0020             0.1683         ONLY: NNVTMCPPMT_PMT_20inch_plate_solid                                   
     36         28         0.0020             0.1687         ONLY: GLw2.equ_up01_FlangeI_Web_FlangeII                                  
     37         95         0.0020             0.1703         ONLY: solidSJReceiverFastern                                              
     38        174         0.0020             0.1706         ONLY: svacSurftube_19V1_0                                                 
     39         73         0.0020             0.1709         ONLY: GZ1.B05_06_FlangeI_Web_FlangeII                                     
     40         54         0.0020             0.1715         ONLY: GLb2.bt03_FlangeI_Web_FlangeII                                      
     41         85         0.0020             0.1717         ONLY: ZC2.B03_B03_FlangeI_Web_FlangeII                                    
     42        175         0.0020             0.1721         ONLY: sSurftube_19V1_0                                                    
     43         92         0.0020             0.1723         ONLY: solidSJCLSanchor                                                    
     44         53         0.0020             0.1725         ONLY: GLb1.bt02_FlangeI_Web_FlangeII                                      
     45         58         0.0020             0.1731         ONLY: GLb1.bt07_FlangeI_Web_FlangeII                                      
     46         65         0.0020             0.1732         ONLY: GZ1.A03_04_FlangeI_Web_FlangeII                                     
     47        124         0.0020             0.1733         ONLY: NNVTMCPPMT_PMT_20inch_inner_solid_head                              
     48          4         0.0020             0.1733         ONLY: Upper_LS_tube                                                       
     49         74         0.0020             0.1735         ONLY: GZ1.B06_07_FlangeI_Web_FlangeII                                     
     50        104         0.0020             0.1735         ONLY: solidXJanchor                                                       
     51         70         0.0020             0.1738         ONLY: GZ1.B02_03_FlangeI_Web_FlangeII                                     
     52         99         0.0020             0.1743         ONLY: sStrut_1                                                            
     53        125         0.0020             0.1744         ONLY: NNVTMCPPMT_PMT_20inch_pmt_solid_head                                
     54         47         0.0020             0.1744         ONLY: GLb1.up04_FlangeI_Web_FlangeII                                      
     55         23         0.0020             0.1745         ONLY: GLw1.up05_up06_FlangeI_Web_FlangeII                                 
     56        102         0.0020             0.1747         ONLY: base_steel                                                          
     57         34         0.0020             0.1747         ONLY: GLw1.bt05_bt06_FlangeI_Web_FlangeII                                 
     58         67         0.0020             0.1748         ONLY: GZ1.A05_06_FlangeI_Web_FlangeII                                     
     59         48         0.0020             0.1751         ONLY: GLb1.up03_FlangeI_Web_FlangeII                                      
     60         49         0.0020             0.1758         ONLY: GLb1.up02_FlangeI_Web_FlangeII                                      
     61         80         0.0020             0.1758         ONLY: ZC2.A02_B03_FlangeI_Web_FlangeII                                    
     62         66         0.0020             0.1763         ONLY: GZ1.A04_05_FlangeI_Web_FlangeII                                     
     63         59         0.0020             0.1764         ONLY: GLb1.bt08_FlangeI_Web_FlangeII                                      
     64         46         0.0020             0.1764         ONLY: GLb1.up05_FlangeI_Web_FlangeII                                      
     65         39         0.0021             0.1770         ONLY: GLw1.bt10_bt11_FlangeI_Web_FlangeII                                 
     66         35         0.0021             0.1770         ONLY: GLw1.bt06_bt07_FlangeI_Web_FlangeII                                 
     67        120         0.0021             0.1787         ONLY: NNVTMCPPMT_PMT_20inch_edge_solid                                    
     68         41         0.0021             0.1792         ONLY: GLb4.up10_FlangeI_Web_FlangeII                                      
     69         61         0.0021             0.1806         ONLY: GLb3.bt10_FlangeI_Web_FlangeII                                      
     70         81         0.0021             0.1813         ONLY: ZC2.A03_B04_FlangeI_Web_FlangeII                                    
     71        301         0.0021             0.1818         ONLY: sWorld                                                              
     72          2         0.0021             0.1828         ONLY: sDomeRockBox                                                        
     73         82         0.0021             0.1832         ONLY: ZC2.A04_B05_FlangeI_Web_FlangeII                                    
     74         27         0.0021             0.1842         ONLY: GLw1.up01_up02_FlangeI_Web_FlangeII                                 
     75         98         0.0021             0.1844         ONLY: sStrut_0                                                            
     76        135         0.0022             0.1857         ONLY: sWaterTube                                                          
     77         22         0.0022             0.1863         ONLY: GLw1.up06_up07_FlangeI_Web_FlangeII                                 
     78         37         0.0022             0.1871         ONLY: GLw1.bt08_bt09_FlangeI_Web_FlangeII                                 
     79        131         0.0022             0.1879         ONLY: PMT_3inch_pmt_solid                                                 
     80          5         0.0022             0.1879         ONLY: Upper_Steel_tube                                                    
     81        128         0.0022             0.1883         ONLY: PMT_3inch_inner2_solid_ell_helper                                   
     82         38         0.0022             0.1889         ONLY: GLw1.bt09_bt10_FlangeI_Web_FlangeII                                 
     83          6         0.0022             0.1896         ONLY: Upper_Tyvek_tube                                                    
     84        129         0.0022             0.1903         ONLY: PMT_3inch_body_solid_ell_ell_helper                                 
     85         21         0.0022             0.1903         ONLY: GLw1.up07_up08_FlangeI_Web_FlangeII                                 
     86         18         0.0022             0.1929         ONLY: GLw1.up10_up11_FlangeI_Web_FlangeII                                 
     87        300         0.0022             0.1932         ONLY: sBottomRock                                                         
     88         17         0.0023             0.1941         ONLY: sTopRock                                                            
     89        117         0.0023             0.1942         ONLY: HamamatsuR12860sMask_virtual                                        
     90         20         0.0023             0.1964         ONLY: GLw1.up08_up09_FlangeI_Web_FlangeII                                 
     91        101         0.0023             0.1975         ONLY: uni1                                                                
     92        296         0.0023             0.1976         ONLY: PMT_20inch_veto_pmt_solid_1_2                                       
     93         12         0.0023             0.1982         ONLY: sPlane                                                              
     94        298         0.0023             0.1987         ONLY: sAirGap                                                             
     95         87         0.0023             0.1999         ONLY: ZC2.A03_A03_FlangeI_Web_FlangeII                                    
     96         36         0.0023             0.2022         ONLY: GLw1.bt07_bt08_FlangeI_Web_FlangeII                                 
     97        137         0.0023             0.2024         ONLY: sSurftube_0V1_0                                                     
     98         50         0.0024             0.2027         ONLY: GLb1.up01_FlangeI_Web_FlangeII                                      
     99         89         0.0024             0.2028         ONLY: sOuterWaterPool                                                     
    100         44         0.0024             0.2036         ONLY: GLb2.up07_FlangeI_Web_FlangeII                                      
    101        103         0.0024             0.2051         ONLY: uni_acrylic1                                                        
    102         56         0.0024             0.2054         ONLY: GLb1.bt05_FlangeI_Web_FlangeII                                      
    103         32         0.0024             0.2062         ONLY: GLw2.bt03_bt04_FlangeI_Web_FlangeII                                 
    104         14         0.0024             0.2068         ONLY: sAirTT                                                              
    105        106         0.0024             0.2071         ONLY: HamamatsuR12860sMask                                                
    106         11         0.0024             0.2078         ONLY: sPanel                                                              
    107        177         0.0024             0.2103         ONLY: sSurftube_20V1_0                                                    
    108        116         0.0024             0.2113         ONLY: HamamatsuR12860_PMT_20inch_pmt_solid_1_4                            
    109         15         0.0025             0.2115         ONLY: sExpHall                                                            
    110          8         0.0025             0.2116         ONLY: sBar_0                                                              
    111         76         0.0025             0.2128         ONLY: ZC2.A03_B03_FlangeI_Web_FlangeII                                    
    112         10         0.0025             0.2139         ONLY: sPanelTape                                                          
    113         57         0.0025             0.2142         ONLY: GLb1.bt06_FlangeI_Web_FlangeII                                      
    114          9         0.0025             0.2152         ONLY: sBar_1                                                              
    115         63         0.0025             0.2181         ONLY: GZ1.A01_02_FlangeI_Web_FlangeII                                     
    116        176         0.0026             0.2204         ONLY: svacSurftube_20V1_0                                                 
    117        254         0.0026             0.2209         ONLY: svacSurftube_20V1_1                                                 
    118         86         0.0026             0.2220         ONLY: ZC2.B05_B05_FlangeI_Web_FlangeII                                    
    119         68         0.0026             0.2233         ONLY: GZ1.A06_07_FlangeI_Web_FlangeII                                     
    120         83         0.0026             0.2240         ONLY: ZC2.A05_B06_FlangeI_Web_FlangeII                                    
    121        253         0.0026             0.2269         ONLY: sSurftube_19V1_1                                                    
    122         60         0.0027             0.2311         ONLY: GLb3.bt09_FlangeI_Web_FlangeII                                      
    123        136         0.0027             0.2316         ONLY: svacSurftube_0V1_0                                                  
    124        173         0.0027             0.2327         ONLY: sSurftube_18V1_0                                                    
    125        258         0.0027             0.2337         ONLY: svacSurftube_22V1_1                                                 
    126        105         0.0027             0.2351         ONLY: solidXJfixture                                                      
    127         13         0.0027             0.2365         ONLY: sWall                                                               
    128        214         0.0028             0.2389         ONLY: svacSurftube_0V1_1                                                  
    129         42         0.0028             0.2396         ONLY: GLb3.up09_FlangeI_Web_FlangeII                                      
    130        255         0.0029             0.2469         ONLY: sSurftube_20V1_1                                                    
    131        293         0.0029             0.2472         ONLY: sReflectorInCD                                                      
    132        132         0.0029             0.2475         ONLY: sChimneyAcrylic                                                     
    133        251         0.0029             0.2476         ONLY: sSurftube_18V1_1                                                    
    134        250         0.0029             0.2517         ONLY: svacSurftube_18V1_1                                                 
    135        126         0.0029             0.2531         ONLY: NNVTMCPPMTsMask_virtual                                             
    136         30         0.0031             0.2640         ONLY: GLw3.bt01_bt02_FlangeI_Web_FlangeII                                 
    137        292         0.0031             0.2650         ONLY: sInnerWater                                                         
    138        247         0.0031             0.2665         ONLY: sSurftube_16V1_1                                                    
    139        259         0.0031             0.2677         ONLY: sSurftube_22V1_1                                                    
    140          1         0.0031             0.2695         ONLY: sTopRock_dome                                                       
    141          0         0.0031             0.2709         ONLY: sTopRock_domeAir                                                    
    142        289         0.0032             0.2722         ONLY: sSurftube_37V1_1                                                    
    143        299         0.0032             0.2774         ONLY: sPoolLining                                                         
    144        169         0.0032             0.2792         ONLY: sSurftube_16V1_0                                                    
    145        181         0.0033             0.2804         ONLY: sSurftube_22V1_0                                                    
    146        118         0.0033             0.2827         ONLY: NNVTMCPPMTsMask                                                     
    147        262         0.0033             0.2829         ONLY: svacSurftube_24V1_1                                                 
    148        246         0.0033             0.2829         ONLY: svacSurftube_16V1_1                                                 
    149        291         0.0033             0.2878         ONLY: sSurftube_38V1_1                                                    
    150        134         0.0034             0.2898         ONLY: sChimneySteel                                                       
    151        217         0.0034             0.2912         ONLY: sSurftube_1V1_1                                                     
    152        180         0.0034             0.2925         ONLY: svacSurftube_22V1_0                                                 
    153         69         0.0034             0.2951         ONLY: GZ1.B01_02_FlangeI_Web_FlangeII                                     
    154        168         0.0035             0.2992         ONLY: svacSurftube_16V1_0                                                 
    155        216         0.0035             0.3005         ONLY: svacSurftube_1V1_1                                                  
    156         93         0.0035             0.3029         ONLY: solidSJFixture                                                      
    157        288         0.0035             0.3043         ONLY: svacSurftube_37V1_1                                                 
    158        243         0.0037             0.3232         ONLY: sSurftube_14V1_1                                                    
    159        263         0.0038             0.3246         ONLY: sSurftube_24V1_1                                                    
    160        107         0.0038             0.3313         ONLY: HamamatsuR12860Tail                                                 
    161        242         0.0039             0.3345         ONLY: svacSurftube_14V1_1                                                 
    162        171         0.0039             0.3352         ONLY: sSurftube_17V1_0                                                    
    163        178         0.0039             0.3365         ONLY: svacSurftube_21V1_0                                                 
    164        290         0.0039             0.3380         ONLY: svacSurftube_38V1_1                                                 
    165        165         0.0039             0.3398         ONLY: sSurftube_14V1_0                                                    
    166        256         0.0039             0.3401         ONLY: svacSurftube_21V1_1                                                 
    167        248         0.0040             0.3434         ONLY: svacSurftube_17V1_1                                                 
    168        185         0.0040             0.3488         ONLY: sSurftube_24V1_0                                                    
    169        164         0.0041             0.3495         ONLY: svacSurftube_14V1_0                                                 
    170        212         0.0041             0.3505         ONLY: svacSurftube_38V1_0                                                 
    171        184         0.0041             0.3512         ONLY: svacSurftube_24V1_0                                                 
    172        170         0.0041             0.3517         ONLY: svacSurftube_17V1_0                                                 
    173        249         0.0041             0.3552         ONLY: sSurftube_17V1_1                                                    
    174        115         0.0042             0.3610         ONLY: HamamatsuR12860_PMT_20inch_inner_solid_1_4                          
    175        284         0.0042             0.3627         ONLY: svacSurftube_35V1_1                                                 
    176        213         0.0043             0.3692         ONLY: sSurftube_38V1_0                                                    
    177        239         0.0044             0.3838         ONLY: sSurftube_12V1_1                                                    
    178        267         0.0045             0.3865         ONLY: sSurftube_26V1_1                                                    
    179        220         0.0045             0.3890         ONLY: svacSurftube_3V1_1                                                  
    180        285         0.0045             0.3899         ONLY: sSurftube_35V1_1                                                    
    181         16         0.0046             0.3937         ONLY: sExpRockBox                                                         
    182        266         0.0048             0.4119         ONLY: svacSurftube_26V1_1                                                 
    183        257         0.0049             0.4201         ONLY: sSurftube_21V1_1                                                    
    184        188         0.0049             0.4267         ONLY: svacSurftube_26V1_0                                                 
    185        160         0.0050             0.4274         ONLY: svacSurftube_12V1_0                                                 
    186        161         0.0050             0.4334         ONLY: sSurftube_12V1_0                                                    
    187        215         0.0050             0.4349         ONLY: sSurftube_0V1_1                                                     
    188        271         0.0051             0.4408         ONLY: sSurftube_28V1_1                                                    
    189        238         0.0051             0.4411         ONLY: svacSurftube_12V1_1                                                 
    190        221         0.0051             0.4432         ONLY: sSurftube_3V1_1                                                     
    191        183         0.0052             0.4446         ONLY: sSurftube_23V1_0                                                    
    192        166         0.0052             0.4475         ONLY: svacSurftube_15V1_0                                                 
    193        235         0.0053             0.4530         ONLY: sSurftube_10V1_1                                                    
    194        167         0.0053             0.4536         ONLY: sSurftube_15V1_0                                                    
    195        270         0.0053             0.4542         ONLY: svacSurftube_28V1_1                                                 
    196         88         0.0053             0.4564         ONLY: ZC2.A05_A05_FlangeI_Web_FlangeII                                    
    197        182         0.0053             0.4571         ONLY: svacSurftube_23V1_0                                                 
    198        211         0.0053             0.4613         ONLY: sSurftube_37V1_0                                                    
    199        234         0.0054             0.4645         ONLY: svacSurftube_10V1_1                                                 
    200        139         0.0054             0.4686         ONLY: sSurftube_1V1_0                                                     
    201        138         0.0054             0.4689         ONLY: svacSurftube_1V1_0                                                  
    202         19         0.0054             0.4689         ONLY: GLw1.up09_up10_FlangeI_Web_FlangeII                                 
    203        231         0.0055             0.4735         ONLY: sSurftube_8V1_1                                                     
    204        274         0.0055             0.4747         ONLY: svacSurftube_30V1_1                                                 
    205          7         0.0055             0.4778         ONLY: Upper_Chimney                                                       
    206        294         0.0055             0.4784         ONLY: mask_PMT_20inch_vetosMask                                           
    207        218         0.0056             0.4809         ONLY: svacSurftube_2V1_1                                                  
    208         62         0.0056             0.4810         ONLY: GLb3.bt11_FlangeI_Web_FlangeII                                      
    209         40         0.0056             0.4830         ONLY: GLb3.up11_FlangeI_Web_FlangeII                                      
    210         96         0.0056             0.4866         ONLY: sTarget                                                             
    211        260         0.0057             0.4904         ONLY: svacSurftube_23V1_1                                                 
    212         24         0.0057             0.4927         ONLY: GLw1.up04_up05_FlangeI_Web_FlangeII                                 
    213        261         0.0057             0.4952         ONLY: sSurftube_23V1_1                                                    
    214        230         0.0058             0.4967         ONLY: svacSurftube_8V1_1                                                  
    215        133         0.0058             0.4978         ONLY: sChimneyLS                                                          
    216        192         0.0058             0.5038         ONLY: svacSurftube_28V1_0                                                 
    217        210         0.0059             0.5056         ONLY: svacSurftube_37V1_0                                                 
    218        157         0.0059             0.5085         ONLY: sSurftube_10V1_0                                                    
    219        245         0.0059             0.5099         ONLY: sSurftube_15V1_1                                                    
    220        241         0.0059             0.5127         ONLY: sSurftube_13V1_1                                                    
    221        219         0.0060             0.5147         ONLY: sSurftube_2V1_1                                                     
    222        119         0.0060             0.5150         ONLY: NNVTMCPPMTTail                                                      
    223        265         0.0060             0.5151         ONLY: sSurftube_25V1_1                                                    
    224        142         0.0060             0.5168         ONLY: svacSurftube_3V1_0                                                  
    225        264         0.0061             0.5221         ONLY: svacSurftube_25V1_1                                                 
    226        244         0.0061             0.5237         ONLY: svacSurftube_15V1_1                                                 
    227        156         0.0061             0.5261         ONLY: svacSurftube_10V1_0                                                 
    228         77         0.0061             0.5282         ONLY: ZC2.A04_B04_FlangeI_Web_FlangeII                                    
    229        287         0.0061             0.5294         ONLY: sSurftube_36V1_1                                                    
    230         29         0.0062             0.5346         ONLY: GLw2.equ_bt01_FlangeI_Web_FlangeII                                  
    231        283         0.0063             0.5407         ONLY: sSurftube_34V1_1                                                    
    232        275         0.0063             0.5407         ONLY: sSurftube_30V1_1                                                    
    233        143         0.0064             0.5503         ONLY: sSurftube_3V1_0                                                     
    234        172         0.0064             0.5529         ONLY: svacSurftube_18V1_0                                                 
    235        206         0.0064             0.5534         ONLY: svacSurftube_35V1_0                                                 
    236        162         0.0064             0.5548         ONLY: svacSurftube_13V1_0                                                 
    237        223         0.0065             0.5584         ONLY: sSurftube_4V1_1                                                     
    238        222         0.0065             0.5584         ONLY: svacSurftube_4V1_1                                                  
    239        186         0.0066             0.5727         ONLY: svacSurftube_25V1_0                                                 
    240        282         0.0067             0.5749         ONLY: svacSurftube_34V1_1                                                 
    241        152         0.0068             0.5835         ONLY: svacSurftube_8V1_0                                                  
    242         97         0.0069             0.5911         ONLY: sAcrylic                                                            
    243        196         0.0069             0.5977         ONLY: svacSurftube_30V1_0                                                 
    244        197         0.0071             0.6093         ONLY: sSurftube_30V1_0                                                    
    245        163         0.0073             0.6264         ONLY: sSurftube_13V1_0                                                    
    246        179         0.0073             0.6321         ONLY: sSurftube_21V1_0                                                    
    247        207         0.0074             0.6355         ONLY: sSurftube_35V1_0                                                    
    248        280         0.0079             0.6777         ONLY: svacSurftube_33V1_1                                                 
    249        189         0.0079             0.6850         ONLY: sSurftube_26V1_0                                                    
    250        281         0.0083             0.7121         ONLY: sSurftube_33V1_1                                                    
    251        224         0.0085             0.7306         ONLY: svacSurftube_5V1_1                                                  
    252        190         0.0089             0.7636         ONLY: svacSurftube_27V1_0                                                 
    253        159         0.0089             0.7660         ONLY: sSurftube_11V1_0                                                    
    254        279         0.0089             0.7672         ONLY: sSurftube_32V1_1                                                    
    255        227         0.0091             0.7830         ONLY: sSurftube_6V1_1                                                     
    256        225         0.0091             0.7837         ONLY: sSurftube_5V1_1                                                     
    257        193         0.0091             0.7856         ONLY: sSurftube_28V1_0                                                    
    258        277         0.0093             0.8051         ONLY: sSurftube_31V1_1                                                    
    259        158         0.0094             0.8072         ONLY: svacSurftube_11V1_0                                                 
    260        226         0.0094             0.8081         ONLY: svacSurftube_6V1_1                                                  
    261        278         0.0094             0.8127         ONLY: svacSurftube_32V1_1                                                 
    262        240         0.0098             0.8431         ONLY: svacSurftube_13V1_1                                                 
    263        204         0.0098             0.8471         ONLY: svacSurftube_34V1_0                                                 
    264        229         0.0100             0.8583         ONLY: sSurftube_7V1_1                                                     
    265        286         0.0101             0.8698         ONLY: svacSurftube_36V1_1                                                 
    266        187         0.0102             0.8769         ONLY: sSurftube_25V1_0                                                    
    267        269         0.0102             0.8829         ONLY: sSurftube_27V1_1                                                    
    268        140         0.0102             0.8834         ONLY: svacSurftube_2V1_0                                                  
    269        208         0.0103             0.8915         ONLY: svacSurftube_36V1_0                                                 
    270        141         0.0104             0.8976         ONLY: sSurftube_2V1_0                                                     
    271        233         0.0104             0.8982         ONLY: sSurftube_9V1_1                                                     
    272        228         0.0106             0.9114         ONLY: svacSurftube_7V1_1                                                  
    273        273         0.0108             0.9357         ONLY: sSurftube_29V1_1                                                    
    274        147         0.0110             0.9460         ONLY: sSurftube_5V1_0                                                     
    275        200         0.0110             0.9497         ONLY: svacSurftube_32V1_0                                                 
    276        202         0.0111             0.9543         ONLY: svacSurftube_33V1_0                                                 
    277        148         0.0111             0.9548         ONLY: svacSurftube_6V1_0                                                  
    278        203         0.0111             0.9557         ONLY: sSurftube_33V1_0                                                    
    279        151         0.0112             0.9619         ONLY: sSurftube_7V1_0                                                     
    280        149         0.0112             0.9649         ONLY: sSurftube_6V1_0                                                     
    281        232         0.0112             0.9661         ONLY: svacSurftube_9V1_1                                                  
    282        150         0.0112             0.9665         ONLY: svacSurftube_7V1_0                                                  
    283        194         0.0112             0.9690         ONLY: svacSurftube_29V1_0                                                 
    284        154         0.0113             0.9737         ONLY: svacSurftube_9V1_0                                                  
    285        155         0.0113             0.9738         ONLY: sSurftube_9V1_0                                                     
    286        146         0.0114             0.9819         ONLY: svacSurftube_5V1_0                                                  
    287          t         0.0116             1.0000         ALL                                                                       
    288        153         0.0117             1.0118         ONLY: sSurftube_8V1_0                                                     
    289        199         0.0120             1.0343         ONLY: sSurftube_31V1_0                                                    
    290        236         0.0120             1.0359         ONLY: svacSurftube_11V1_1                                                 
    291        195         0.0121             1.0469         ONLY: sSurftube_29V1_0                                                    
    292        191         0.0122             1.0535         ONLY: sSurftube_27V1_0                                                    
    293        272         0.0124             1.0713         ONLY: svacSurftube_29V1_1                                                 
    294        268         0.0127             1.0953         ONLY: svacSurftube_27V1_1                                                 
    295        209         0.0134             1.1580         ONLY: sSurftube_36V1_0                                                    
    296        276         0.0135             1.1663         ONLY: svacSurftube_31V1_1                                                 
    297        237         0.0136             1.1698         ONLY: sSurftube_11V1_1                                                    
    298        205         0.0137             1.1798         ONLY: sSurftube_34V1_0                                                    
    299        144         0.0138             1.1883         ONLY: svacSurftube_4V1_0                                                  
    300        198         0.0138             1.1920         ONLY: svacSurftube_31V1_0                                                 
    301        145         0.0144             1.2394         ONLY: sSurftube_4V1_0                                                     
    302        201         0.0149             1.2888         ONLY: sSurftube_32V1_0                                                    
    idx         -e        time(s)           relative         enabled geometry description  



ELV scanning with cxr_view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



::

    P[blyth@localhost opticks]$ ~/o/CSGOptiX/cxr_scan_elv.sh
                       0 : /home/blyth/o/CSGOptiX/cxr_scan_elv.sh 
             BASH_SOURCE : /home/blyth/o/CSGOptiX/cxr_scan_elv.sh 
                    GEOM : J_2024aug27 
                     cfd : /home/blyth/.opticks/GEOM/J_2024aug27/CSGFoundry 
                     nmm : 10 
                     nlv : 301 
                     NMM : 10 
                     NLV : 301 
                  script : cxr_view 
                  SCRIPT : cxr_view 
                thisname : cxr_scan_elv.sh 
                thisstem : cxr_scan_elv 
                    scan : scan-elv 
                    SCAN : scan-elv 
                    vars : 0 BASH_SOURCE GEOM cfd nmm nlv NMM NLV script SCRIPT thisname thisstem scan SCAN vars 
    2024-09-12 22:12:18.932 INFO  [96819] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0125 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:12:22.287 INFO  [96846] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t0_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0119 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:12:25.460 INFO  [96874] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t1_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0121 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:12:28.959 INFO  [96900] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t2_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0125 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:12:32.408 INFO  [96927] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t3_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0144 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:12:35.771 INFO  [96955] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t4_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0161 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:12:39.098 INFO  [96981] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t5_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0169 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:12:42.493 INFO  [97008] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t6_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0152 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:12:45.796 INFO  [97035] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t7_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:12:49.185 INFO  [97061] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t8_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:12:52.475 INFO  [97088] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t9_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:12:55.740 INFO  [97116] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t10_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:12:59.084 INFO  [97142] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t11_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:13:02.870 INFO  [97177] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t12_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:13:06.451 INFO  [97205] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t13_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:13:10.193 INFO  [97232] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t14_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:13:13.476 INFO  [97259] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t15_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0156 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:13:16.669 INFO  [97286] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t16_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0137 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:13:19.921 INFO  [97312] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t17_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0141 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:13:23.175 INFO  [97339] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t18_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:13:26.458 INFO  [97365] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t19_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:13:29.809 INFO  [97393] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t20_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0135 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:13:33.126 INFO  [97420] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t21_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0132 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:13:36.367 INFO  [97446] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t22_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0126 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:13:39.613 INFO  [97473] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t23_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:13:42.897 INFO  [97500] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t24_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:13:46.213 INFO  [97527] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t25_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:13:49.583 INFO  [97554] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t26_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:13:52.844 INFO  [97580] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t27_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0126 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:13:56.126 INFO  [97607] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t28_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0125 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:13:59.783 INFO  [97642] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t29_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0133 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:14:03.503 INFO  [97669] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t30_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:14:06.995 INFO  [97696] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t31_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0137 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:14:10.419 INFO  [97723] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t32_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:14:13.846 INFO  [97750] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t33_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0185 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:14:17.084 INFO  [97777] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t34_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0164 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:14:20.350 INFO  [97803] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t35_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:14:23.557 INFO  [97830] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t36_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:14:26.896 INFO  [97857] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t37_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:14:30.599 INFO  [97884] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t38_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:14:34.497 INFO  [97912] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t39_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:14:38.026 INFO  [97938] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t40_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:14:41.322 INFO  [97965] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t41_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0125 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:14:44.700 INFO  [97993] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t42_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0157 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:14:48.010 INFO  [98020] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t43_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:14:51.251 INFO  [98046] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t44_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:14:54.526 INFO  [98073] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t45_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:14:57.753 INFO  [98100] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t46_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0137 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:15:01.008 INFO  [98135] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t47_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0151 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:15:04.429 INFO  [98162] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t48_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0134 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:15:07.631 INFO  [98188] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t49_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:15:10.825 INFO  [98215] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t50_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:15:14.097 INFO  [98241] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t51_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:15:17.396 INFO  [98280] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t52_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0144 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:15:20.769 INFO  [98309] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t53_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:15:24.020 INFO  [98337] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t54_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0132 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:15:27.263 INFO  [98369] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t55_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0139 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:15:30.524 INFO  [98408] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t56_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0136 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:15:33.780 INFO  [98435] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t57_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0143 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:15:37.096 INFO  [98462] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t58_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:15:40.406 INFO  [98489] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t59_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:15:43.789 INFO  [98515] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t60_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0140 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:15:47.190 INFO  [98542] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t61_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:15:50.569 INFO  [98569] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t62_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0132 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:15:53.819 INFO  [98596] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t63_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0139 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:15:57.079 INFO  [98622] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t64_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:16:00.576 INFO  [98658] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t65_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:16:03.967 INFO  [98685] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t66_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0135 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:16:07.490 INFO  [98712] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t67_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:16:10.730 INFO  [98739] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t68_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0147 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:16:13.926 INFO  [98766] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t69_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:16:17.193 INFO  [98792] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t70_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0126 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:16:20.474 INFO  [98819] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t71_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:16:23.700 INFO  [98846] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t72_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0160 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:16:27.034 INFO  [98872] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t73_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0132 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:16:30.361 INFO  [98899] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t74_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:16:33.925 INFO  [98926] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t75_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0148 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:16:37.203 INFO  [98952] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t76_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:16:40.717 INFO  [98980] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t77_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:16:44.080 INFO  [99007] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t78_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0125 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:16:47.693 INFO  [99034] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t79_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:16:51.120 INFO  [99060] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t80_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0134 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:16:54.402 INFO  [99087] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t81_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:16:57.856 INFO  [99114] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t82_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0134 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:17:01.181 INFO  [99148] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t83_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0163 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:17:04.528 INFO  [99175] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t84_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0157 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:17:08.020 INFO  [99202] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t85_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0134 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:17:11.507 INFO  [99229] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t86_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:17:14.844 INFO  [99256] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t87_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:17:18.096 INFO  [99283] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t88_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:17:21.515 INFO  [99310] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t89_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0125 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:17:24.862 INFO  [99336] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t90_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0126 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:17:28.147 INFO  [99364] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t91_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0124 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:17:31.395 INFO  [99391] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t92_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:17:34.792 INFO  [99417] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t93_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0136 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:17:38.027 INFO  [99444] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t94_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:17:41.363 INFO  [99471] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t95_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:17:44.622 INFO  [99497] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t96_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:17:47.923 INFO  [99524] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t97_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0125 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:17:51.104 INFO  [99550] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t98_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0154 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:17:54.348 INFO  [99578] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t99_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:17:57.704 INFO  [99605] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t100_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0134 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:18:01.047 INFO  [99639] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t101_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0163 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:18:04.253 INFO  [99666] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t102_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0113 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:18:07.437 INFO  [99694] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t103_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0143 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:18:10.747 INFO  [99720] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t104_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:18:14.052 INFO  [99747] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t105_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0135 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:18:17.337 INFO  [99773] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t106_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0126 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:18:20.629 INFO  [99800] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t107_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0125 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:18:23.925 INFO  [99827] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t108_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:18:27.139 INFO  [99853] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t109_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0133 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:18:30.385 INFO  [99881] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t110_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:18:33.628 INFO  [99907] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t111_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:18:37.045 INFO  [99934] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t112_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0132 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:18:40.524 INFO  [99961] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t113_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:18:43.940 INFO  [99989] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t114_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:18:47.164 INFO  [100015] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t115_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:18:50.592 INFO  [100048] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t116_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0122 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:18:53.867 INFO  [100081] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t117_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0143 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:18:57.114 INFO  [100107] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t118_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0115 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:19:00.479 INFO  [100142] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t119_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0123 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:19:03.749 INFO  [100169] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t120_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:19:07.010 INFO  [100195] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t121_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0170 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:19:10.232 INFO  [100223] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t122_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0134 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:19:13.506 INFO  [100249] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t123_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:19:16.727 INFO  [100276] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t124_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0121 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:19:19.982 INFO  [100303] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t125_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0114 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:19:23.240 INFO  [100329] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t126_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0117 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:19:26.526 INFO  [100356] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t127_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0156 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:19:29.826 INFO  [100382] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t128_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:19:33.065 INFO  [100409] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t129_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0126 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:19:36.334 INFO  [100436] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t130_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0136 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:19:39.542 INFO  [100462] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t131_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0113 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:19:42.729 INFO  [100489] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t132_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:19:45.950 INFO  [100516] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t133_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0132 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:19:49.149 INFO  [100543] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t134_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:19:51.865 ERROR [100569] [CSGFoundry::getMeshPrim@2242]  midx 135 mord 0 select_prim.size 0 mord_in_range 0
    2024-09-12 22:19:51.865 FATAL [100569] [CSGTarget::getInstanceTransform@460] Foundry::getMeshPrim failed for (midx mord) (135 0)
    2024-09-12 22:19:51.865 FATAL [100569] [CSGTarget::getGlobalCenterExtent@345]  failed to get InstanceTransform (midx mord gord) (135 0 0)
    2024-09-12 22:19:51.865 FATAL [100569] [CSGTarget::getGlobalCenterExtent@348]  failed Tran<double>::Invert 
    2024-09-12 22:19:51.865 ERROR [100569] [CSGFoundry::getFrame@3510] Failed to lookup frame with frs [sWaterTube] looks_like_moi 1
    2024-09-12 22:19:51.865 ERROR [100569] [CSGFoundry::getFrame@3452]  frs sWaterTube

    CSGFoundry::getFrame_NOTES
    ===========================

    When CSGFoundry::getFrame fails due to the MOI/FRS string used to target 
    a volume of the geometry failing to find the targetted volume 
    it is usually due to the spec not being appropriate for the geometry. 
    ...


    2024-09-12 22:19:54.813 INFO  [100589] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t136_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0165 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:19:58.125 INFO  [100617] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t137_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:20:01.378 INFO  [100652] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t138_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0126 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:20:04.635 INFO  [100700] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t139_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0126 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:20:07.796 INFO  [100732] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t140_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0134 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:20:10.999 INFO  [100758] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t141_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0136 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:20:14.243 INFO  [100785] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t142_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:20:17.491 INFO  [100824] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t143_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:20:20.688 INFO  [100850] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t144_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:20:23.843 INFO  [100878] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t145_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0125 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:20:27.008 INFO  [100904] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t146_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:20:30.173 INFO  [100931] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t147_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:20:33.352 INFO  [100957] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t148_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:20:36.547 INFO  [100984] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t149_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:20:39.772 INFO  [101011] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t150_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0138 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:20:43.034 INFO  [101037] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t151_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:20:46.353 INFO  [101064] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t152_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:20:49.958 INFO  [101091] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t153_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:20:53.387 INFO  [101117] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t154_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:20:56.654 INFO  [101144] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t155_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0126 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:21:00.298 INFO  [101180] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t156_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:21:03.815 INFO  [101207] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t157_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:21:07.029 INFO  [101234] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t158_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:21:10.310 INFO  [101261] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t159_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:21:13.530 INFO  [101289] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t160_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:21:16.796 INFO  [101315] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t161_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0168 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:21:20.027 INFO  [101342] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t162_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:21:23.430 INFO  [101369] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t163_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:21:26.797 INFO  [101395] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t164_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:21:30.044 INFO  [101422] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t165_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0132 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:21:33.307 INFO  [101448] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t166_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0150 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:21:36.598 INFO  [101475] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t167_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0126 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:21:39.888 INFO  [101503] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t168_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:21:43.128 INFO  [101529] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t169_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0158 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:21:46.339 INFO  [101556] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t170_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:21:49.585 INFO  [101582] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t171_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:21:52.894 INFO  [101609] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t172_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0125 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:21:56.124 INFO  [101636] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t173_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:21:59.300 INFO  [101662] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t174_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:22:02.517 INFO  [101697] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t175_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0134 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:22:05.697 INFO  [101723] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t176_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0124 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:22:08.881 INFO  [101750] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t177_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:22:12.278 INFO  [101776] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t178_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:22:15.449 INFO  [101804] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t179_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:22:18.680 INFO  [101831] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t180_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0125 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:22:21.935 INFO  [101857] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t181_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:22:25.146 INFO  [101884] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t182_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:22:28.316 INFO  [101911] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t183_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0126 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:22:31.473 INFO  [101938] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t184_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:22:34.661 INFO  [101965] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t185_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:22:37.863 INFO  [101991] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t186_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:22:41.097 INFO  [102018] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t187_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:22:44.524 INFO  [102044] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t188_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0126 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:22:48.252 INFO  [102071] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t189_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:22:51.623 INFO  [102098] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t190_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0125 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:22:54.885 INFO  [102126] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t191_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0165 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:22:58.082 INFO  [102152] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t192_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0137 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:23:01.322 INFO  [102187] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t193_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:23:04.606 INFO  [102213] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t194_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:23:08.050 INFO  [102240] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t195_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:23:11.428 INFO  [102267] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t196_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:23:14.934 INFO  [102294] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t197_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:23:18.243 INFO  [102320] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t198_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0134 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:23:21.708 INFO  [102347] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t199_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0166 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:23:25.068 INFO  [102374] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t200_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0132 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:23:28.336 INFO  [102400] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t201_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:23:31.556 INFO  [102428] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t202_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:23:34.783 INFO  [102455] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t203_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:23:38.033 INFO  [102481] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t204_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:23:41.422 INFO  [102509] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t205_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0159 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:23:44.790 INFO  [102535] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t206_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:23:47.994 INFO  [102562] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t207_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:23:51.318 INFO  [102589] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t208_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:23:54.796 INFO  [102615] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t209_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:23:58.361 INFO  [102642] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t210_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:24:01.621 INFO  [102677] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t211_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:24:04.836 INFO  [102704] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t212_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:24:08.073 INFO  [102730] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t213_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:24:11.284 INFO  [102758] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t214_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:24:14.599 INFO  [102784] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t215_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:24:17.797 INFO  [102836] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t216_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:24:20.987 INFO  [102863] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t217_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0125 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:24:24.184 INFO  [102889] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t218_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0126 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:24:27.413 INFO  [102917] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t219_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:24:30.568 INFO  [102943] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t220_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:24:33.755 INFO  [102970] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t221_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:24:36.936 INFO  [103009] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t222_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:24:40.168 INFO  [103038] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t223_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:24:43.439 INFO  [103065] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t224_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0132 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:24:46.652 INFO  [103092] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t225_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0132 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:24:49.885 INFO  [103119] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t226_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0132 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:24:53.138 INFO  [103156] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t227_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:24:56.448 INFO  [103184] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t228_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:24:59.687 INFO  [103211] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t229_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:25:03.228 INFO  [103246] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t230_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0126 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:25:06.715 INFO  [103272] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t231_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0140 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:25:09.970 INFO  [103299] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t232_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0132 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:25:13.190 INFO  [103326] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t233_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:25:16.433 INFO  [103352] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t234_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0126 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:25:19.763 INFO  [103379] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t235_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0139 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:25:23.027 INFO  [103405] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t236_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:25:26.425 INFO  [103433] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t237_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:25:29.749 INFO  [103460] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t238_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:25:32.904 INFO  [103486] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t239_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:25:36.316 INFO  [103514] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t240_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:25:39.523 INFO  [103541] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t241_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:25:42.814 INFO  [103567] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t242_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:25:46.127 INFO  [103594] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t243_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:25:49.437 INFO  [103620] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t244_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:25:52.602 INFO  [103647] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t245_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:25:55.819 INFO  [103674] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t246_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0132 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:25:59.002 INFO  [103701] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t247_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0126 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:26:02.184 INFO  [103736] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t248_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0139 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:26:05.398 INFO  [103762] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t249_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:26:08.562 INFO  [103802] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t250_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:26:11.726 INFO  [103829] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t251_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:26:14.867 INFO  [103856] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t252_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:26:18.036 INFO  [103883] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t253_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:26:21.319 INFO  [103909] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t254_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:26:24.461 INFO  [103936] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t255_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:26:27.695 INFO  [103962] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t256_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0139 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:26:30.957 INFO  [103989] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t257_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:26:34.193 INFO  [104016] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t258_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0132 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:26:37.552 INFO  [104042] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t259_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:26:40.859 INFO  [104070] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t260_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:26:43.988 INFO  [104096] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t261_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0132 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:26:47.121 INFO  [104123] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t262_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:26:50.263 INFO  [104149] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t263_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0132 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:26:53.396 INFO  [104176] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t264_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:26:56.530 INFO  [104202] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t265_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:26:59.695 INFO  [104229] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t266_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0132 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:27:03.508 INFO  [104264] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t267_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0142 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:27:06.939 INFO  [104291] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t268_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:27:10.171 INFO  [104317] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t269_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:27:13.283 INFO  [104344] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t270_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:27:16.652 INFO  [104371] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t271_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0126 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:27:19.846 INFO  [104398] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t272_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0133 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:27:22.998 INFO  [104425] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t273_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0132 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:27:26.206 INFO  [104452] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t274_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:27:29.410 INFO  [104479] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t275_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:27:32.706 INFO  [104505] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t276_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:27:35.922 INFO  [104532] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t277_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:27:39.045 INFO  [104558] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t278_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:27:42.188 INFO  [104585] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t279_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:27:45.281 INFO  [104611] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t280_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:27:48.495 INFO  [104638] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t281_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:27:51.759 INFO  [104665] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t282_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:27:55.103 INFO  [104692] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t283_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:27:58.635 INFO  [104719] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t284_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0132 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:28:02.092 INFO  [104754] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t285_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:28:05.258 INFO  [104780] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t286_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:28:08.542 INFO  [104807] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t287_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:28:11.730 INFO  [104834] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t288_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:28:14.985 INFO  [104860] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t289_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0127 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:28:18.355 INFO  [104887] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t290_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:28:21.668 INFO  [104913] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t291_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0132 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:28:25.123 INFO  [104940] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t292_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0121 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:28:28.314 INFO  [104967] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t293_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:28:31.510 INFO  [104994] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t294_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0129 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:28:34.812 INFO  [105021] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t295_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0130 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:28:38.093 INFO  [105049] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t296_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:28:41.262 INFO  [105076] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t297_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0128 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:28:44.399 INFO  [105103] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t298_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0123 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:28:47.797 INFO  [105129] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t299_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0124 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:28:50.997 INFO  [105156] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t300_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0123 1:NVIDIA_TITAN_RTX 
    2024-09-12 22:28:54.413 INFO  [105183] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t301_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0131 1:NVIDIA_TITAN_RTX 
    P[blyth@localhost opticks]$ 



::

    A[blyth@localhost CSGOptiX]$ ~/o/CSGOptiX/cxr_scan_elv.sh 
                       0 : /home/blyth/o/CSGOptiX/cxr_scan_elv.sh 
             BASH_SOURCE : /home/blyth/o/CSGOptiX/cxr_scan_elv.sh 
                    GEOM : J_2024aug27 
                     cfd : /home/blyth/.opticks/GEOM/J_2024aug27/CSGFoundry 
                     nmm : 10 
                     nlv : 301 
                     NMM : 10 
                     NLV : 301 
                  script : cxr_view 
                  SCRIPT : cxr_view 
                thisname : cxr_scan_elv.sh 
                thisstem : cxr_scan_elv 
                    scan : scan-elv 
                    SCAN : scan-elv 
                    vars : 0 BASH_SOURCE GEOM cfd nmm nlv NMM NLV script SCRIPT thisname thisstem scan SCAN vars 
    2024-09-12 22:07:57.988 INFO  [36144] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL00000.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:00.297 INFO  [36169] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t0_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0032 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:02.476 INFO  [36193] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t1_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0031 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:04.600 INFO  [36217] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t2_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:06.739 INFO  [36241] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t3_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:08.958 INFO  [36265] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t4_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:11.117 INFO  [36289] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t5_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:13.296 INFO  [36313] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t6_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:15.455 INFO  [36337] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t7_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:17.574 INFO  [36361] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t8_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:19.718 INFO  [36385] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t9_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:21.854 INFO  [36409] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t10_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:24.051 INFO  [36433] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t11_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:26.229 INFO  [36457] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t12_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:28.383 INFO  [36481] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t13_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:30.528 INFO  [36505] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t14_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:32.659 INFO  [36529] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t15_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:34.812 INFO  [36553] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t16_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0037 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:36.981 INFO  [36577] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t17_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:39.131 INFO  [36601] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t18_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:41.283 INFO  [36625] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t19_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:43.425 INFO  [36649] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t20_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:45.591 INFO  [36673] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t21_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:47.768 INFO  [36697] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t22_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:49.900 INFO  [36721] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t23_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:52.046 INFO  [36745] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t24_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:54.211 INFO  [36769] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t25_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:56.609 INFO  [36793] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t26_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:08:59.967 INFO  [36817] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t27_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:02.205 INFO  [36841] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t28_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:04.338 INFO  [36865] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t29_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:06.520 INFO  [36891] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t30_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:08.741 INFO  [36915] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t31_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:10.935 INFO  [36943] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t32_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0035 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:13.139 INFO  [37072] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t33_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:15.322 INFO  [37096] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t34_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:17.518 INFO  [37122] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t35_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:19.656 INFO  [37146] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t36_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0035 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:21.858 INFO  [37170] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t37_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:24.042 INFO  [37215] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t38_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:26.255 INFO  [37239] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t39_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:28.462 INFO  [37279] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t40_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:30.620 INFO  [37303] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t41_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:32.798 INFO  [37327] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t42_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:34.977 INFO  [37351] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t43_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:37.173 INFO  [37396] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t44_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:39.333 INFO  [37420] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t45_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:41.545 INFO  [37447] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t46_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:43.737 INFO  [37471] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t47_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:45.917 INFO  [37495] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t48_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:48.056 INFO  [37519] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t49_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:50.271 INFO  [37543] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t50_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:52.466 INFO  [37567] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t51_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:54.708 INFO  [37591] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t52_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0035 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:09:57.125 INFO  [37615] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t53_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:00.449 INFO  [37639] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t54_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:02.701 INFO  [37663] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t55_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0035 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:04.839 INFO  [37687] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t56_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0035 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:06.979 INFO  [37711] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t57_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:09.199 INFO  [37735] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t58_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:11.409 INFO  [37759] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t59_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:13.563 INFO  [37783] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t60_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:15.767 INFO  [37807] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t61_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:18.008 INFO  [37846] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t62_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:20.183 INFO  [37870] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t63_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:22.395 INFO  [37894] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t64_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:24.618 INFO  [37918] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t65_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:26.818 INFO  [37942] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t66_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:29.012 INFO  [37967] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t67_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:31.214 INFO  [37991] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t68_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:33.442 INFO  [38015] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t69_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:35.623 INFO  [38039] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t70_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0035 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:37.812 INFO  [38063] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t71_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:39.958 INFO  [38087] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t72_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:42.151 INFO  [38111] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t73_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0035 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:44.272 INFO  [38135] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t74_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:46.450 INFO  [38159] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t75_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0035 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:48.636 INFO  [38183] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t76_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0035 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:50.871 INFO  [38207] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t77_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0035 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:53.025 INFO  [38231] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t78_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:55.185 INFO  [38255] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t79_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:10:57.593 INFO  [38279] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t80_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:01.024 INFO  [38303] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t81_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0035 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:03.304 INFO  [38327] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t82_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:05.472 INFO  [38351] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t83_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:07.695 INFO  [38405] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t84_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:09.875 INFO  [38432] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t85_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:12.048 INFO  [38456] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t86_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0035 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:14.251 INFO  [38480] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t87_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0035 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:16.442 INFO  [38504] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t88_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:18.632 INFO  [38528] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t89_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0033 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:20.811 INFO  [38555] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t90_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0033 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:22.982 INFO  [38579] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t91_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0033 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:25.149 INFO  [38603] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t92_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:27.303 INFO  [38627] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t93_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:29.462 INFO  [38655] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t94_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:31.655 INFO  [38679] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t95_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:33.864 INFO  [38703] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t96_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:35.999 INFO  [38727] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t97_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:38.136 INFO  [38754] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t98_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:40.242 INFO  [38778] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t99_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:42.401 INFO  [38803] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t100_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0037 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:44.541 INFO  [38827] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t101_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0037 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:46.684 INFO  [38851] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t102_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:48.815 INFO  [38875] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t103_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0033 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:50.954 INFO  [38899] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t104_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0035 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:53.101 INFO  [38923] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t105_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:55.253 INFO  [38947] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t106_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0033 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:57.592 INFO  [38971] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t107_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:11:59.801 INFO  [38995] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t108_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:03.174 INFO  [39019] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t109_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:05.414 INFO  [39043] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t110_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:07.560 INFO  [39067] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t111_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:09.699 INFO  [39091] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t112_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:11.839 INFO  [39115] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t113_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:14.004 INFO  [39139] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t114_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:16.189 INFO  [39163] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t115_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0033 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:18.366 INFO  [39187] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t116_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0033 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:20.530 INFO  [39211] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t117_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:22.658 INFO  [39235] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t118_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0030 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:24.820 INFO  [39259] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t119_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0033 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:27.007 INFO  [39283] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t120_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:29.187 INFO  [39307] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t121_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:31.387 INFO  [39331] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t122_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:33.628 INFO  [39355] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t123_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:35.830 INFO  [39379] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t124_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0032 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:38.005 INFO  [39403] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t125_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0032 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:40.205 INFO  [39427] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t126_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0031 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:42.427 INFO  [39451] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t127_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:44.605 INFO  [39475] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t128_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:46.767 INFO  [39499] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t129_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:48.979 INFO  [39523] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t130_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:51.151 INFO  [39547] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t131_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0033 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:53.339 INFO  [39571] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t132_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:55.538 INFO  [39595] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t133_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:57.959 INFO  [39620] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t134_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:12:59.986 ERROR [39644] [CSGFoundry::getMeshPrim@2242]  midx 135 mord 0 select_prim.size 0 mord_in_range 0
    2024-09-12 22:12:59.986 FATAL [39644] [CSGTarget::getInstanceTransform@460] Foundry::getMeshPrim failed for (midx mord) (135 0)
    2024-09-12 22:12:59.986 FATAL [39644] [CSGTarget::getGlobalCenterExtent@345]  failed to get InstanceTransform (midx mord gord) (135 0 0)
    2024-09-12 22:12:59.986 FATAL [39644] [CSGTarget::getGlobalCenterExtent@348]  failed Tran<double>::Invert 
    2024-09-12 22:12:59.986 ERROR [39644] [CSGFoundry::getFrame@3510] Failed to lookup frame with frs [sWaterTube] looks_like_moi 1
    2024-09-12 22:12:59.986 ERROR [39644] [CSGFoundry::getFrame@3452]  frs sWaterTube

    CSGFoundry::getFrame_NOTES
    ===========================

    When CSGFoundry::getFrame fails due to the MOI/FRS string used to target 
    ...

    2024-09-12 22:13:03.354 INFO  [39659] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t136_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:05.613 INFO  [39683] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t137_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:07.758 INFO  [39707] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t138_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:09.880 INFO  [39731] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t139_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:12.074 INFO  [39755] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t140_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:14.216 INFO  [39779] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t141_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:16.355 INFO  [39803] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t142_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:18.519 INFO  [39827] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t143_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:20.696 INFO  [39851] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t144_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:22.867 INFO  [39876] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t145_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:25.034 INFO  [39900] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t146_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:27.192 INFO  [39924] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t147_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:29.387 INFO  [39948] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t148_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:31.511 INFO  [39972] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t149_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:33.660 INFO  [39996] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t150_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:35.801 INFO  [40020] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t151_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:37.955 INFO  [40044] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t152_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:40.089 INFO  [40068] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t153_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:42.277 INFO  [40092] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t154_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:44.425 INFO  [40116] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t155_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:46.547 INFO  [40140] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t156_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:48.689 INFO  [40164] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t157_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:50.811 INFO  [40188] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t158_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:52.980 INFO  [40212] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t159_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:55.140 INFO  [40236] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t160_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:57.348 INFO  [40260] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t161_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:13:59.739 INFO  [40284] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t162_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:01.901 INFO  [40308] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t163_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:05.223 INFO  [40332] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t164_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:07.438 INFO  [40356] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t165_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:09.552 INFO  [40380] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t166_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:11.723 INFO  [40405] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t167_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:13.881 INFO  [40430] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t168_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:16.033 INFO  [40454] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t169_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:18.198 INFO  [40478] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t170_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:20.339 INFO  [40502] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t171_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:22.496 INFO  [40526] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t172_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:24.616 INFO  [40550] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t173_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:26.764 INFO  [40574] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t174_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:28.838 INFO  [40598] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t175_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0035 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:30.991 INFO  [40622] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t176_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:33.115 INFO  [40646] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t177_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:35.249 INFO  [40670] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t178_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:37.371 INFO  [40694] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t179_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:39.528 INFO  [40718] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t180_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:41.706 INFO  [40742] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t181_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:43.893 INFO  [40766] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t182_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:46.114 INFO  [40790] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t183_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:48.248 INFO  [40814] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t184_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:50.363 INFO  [40838] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t185_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:52.512 INFO  [40862] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t186_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:54.675 INFO  [40886] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t187_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:56.839 INFO  [40910] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t188_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:14:59.025 INFO  [40934] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t189_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:01.412 INFO  [40958] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t190_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:03.530 INFO  [40982] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t191_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:06.960 INFO  [41006] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t192_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:09.172 INFO  [41030] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t193_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:11.310 INFO  [41054] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t194_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:13.499 INFO  [41078] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t195_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:15.645 INFO  [41102] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t196_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:17.800 INFO  [41126] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t197_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:19.901 INFO  [41150] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t198_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:22.071 INFO  [41174] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t199_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:24.238 INFO  [41198] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t200_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:26.431 INFO  [41222] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t201_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:28.616 INFO  [41247] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t202_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:30.793 INFO  [41271] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t203_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:33.016 INFO  [41295] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t204_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:35.198 INFO  [41319] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t205_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:37.332 INFO  [41343] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t206_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:39.461 INFO  [41367] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t207_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:41.674 INFO  [41391] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t208_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:43.849 INFO  [41415] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t209_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:45.990 INFO  [41439] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t210_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:48.148 INFO  [41463] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t211_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:50.343 INFO  [41487] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t212_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:52.446 INFO  [41511] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t213_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:54.658 INFO  [41535] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t214_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:56.805 INFO  [41559] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t215_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:15:58.925 INFO  [41583] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t216_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:01.070 INFO  [41607] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t217_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:03.446 INFO  [41631] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t218_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:05.598 INFO  [41655] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t219_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:08.984 INFO  [41679] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t220_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:11.264 INFO  [41703] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t221_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:13.422 INFO  [41727] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t222_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:15.560 INFO  [41751] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t223_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:17.662 INFO  [41775] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t224_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:19.797 INFO  [41799] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t225_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:21.924 INFO  [41823] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t226_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:24.087 INFO  [41847] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t227_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:26.217 INFO  [41871] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t228_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:28.319 INFO  [41895] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t229_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:30.504 INFO  [41919] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t230_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:32.713 INFO  [41943] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t231_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:34.898 INFO  [41967] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t232_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:37.115 INFO  [41991] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t233_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:39.280 INFO  [42015] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t234_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:41.457 INFO  [42039] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t235_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:43.613 INFO  [42064] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t236_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:45.774 INFO  [42088] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t237_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:47.972 INFO  [42112] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t238_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:50.189 INFO  [42136] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t239_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:52.405 INFO  [42160] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t240_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:54.569 INFO  [42184] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t241_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:56.729 INFO  [42208] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t242_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:16:58.842 INFO  [42232] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t243_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:00.986 INFO  [42256] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t244_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:03.153 INFO  [42280] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t245_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:05.527 INFO  [42304] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t246_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:07.711 INFO  [42328] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t247_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:11.099 INFO  [42352] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t248_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:13.415 INFO  [42376] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t249_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:15.602 INFO  [42400] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t250_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:17.847 INFO  [42424] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t251_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:20.020 INFO  [42448] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t252_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0035 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:22.138 INFO  [42472] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t253_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:24.248 INFO  [42496] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t254_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:26.416 INFO  [42520] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t255_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:28.587 INFO  [42544] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t256_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:30.701 INFO  [42568] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t257_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:32.842 INFO  [42592] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t258_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:35.015 INFO  [42616] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t259_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:37.159 INFO  [42640] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t260_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:39.355 INFO  [42664] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t261_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:41.495 INFO  [42688] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t262_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:43.692 INFO  [42712] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t263_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:45.918 INFO  [42736] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t264_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:48.094 INFO  [42760] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t265_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:50.273 INFO  [42784] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t266_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:52.475 INFO  [42808] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t267_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:54.673 INFO  [42832] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t268_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:56.859 INFO  [42856] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t269_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:17:59.078 INFO  [42881] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t270_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:01.249 INFO  [42905] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t271_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:03.436 INFO  [42929] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t272_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:05.792 INFO  [42953] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t273_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:07.967 INFO  [42977] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t274_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:11.273 INFO  [43001] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t275_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:13.531 INFO  [43025] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t276_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:15.729 INFO  [43049] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t277_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:17.911 INFO  [43073] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t278_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:20.107 INFO  [43097] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t279_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:22.268 INFO  [43121] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t280_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:24.445 INFO  [43145] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t281_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:26.611 INFO  [43169] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t282_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:28.801 INFO  [43193] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t283_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:31.007 INFO  [43217] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t284_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:33.202 INFO  [43241] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t285_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:35.400 INFO  [43265] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t286_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:37.548 INFO  [43289] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t287_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:39.727 INFO  [43313] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t288_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:41.908 INFO  [43337] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t289_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:44.060 INFO  [43361] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t290_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:46.265 INFO  [43385] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t291_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:48.444 INFO  [43409] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t292_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0031 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:50.698 INFO  [43433] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t293_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0031 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:52.858 INFO  [43457] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t294_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:55.073 INFO  [43481] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t295_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:57.274 INFO  [43505] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t296_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:18:59.492 INFO  [43529] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t297_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0034 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:19:01.678 INFO  [43553] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t298_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0033 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:19:03.888 INFO  [43577] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t299_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0033 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:19:06.294 INFO  [43601] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t300_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0033 0:NVIDIA_RTX_5000_Ada_Generation 
    2024-09-12 22:19:08.513 INFO  [43625] [CSGOptiX::render_save_@1279] /tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD0/80000/sWaterTube/scan-elv/cxr_view_emm_t0_elv_t301_eye_-1,-1,-1,1__zoom_1__tmin_0.4__ALL.jpg :     0.0035 0:NVIDIA_RTX_5000_Ada_Generation 
    A[blyth@localhost CSGOptiX]$ 


