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



