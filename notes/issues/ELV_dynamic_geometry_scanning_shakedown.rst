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




ana shakedown : descriptions are all "ALL" 
--------------------------------------------

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




