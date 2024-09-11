ELV_dynamic_geometry_scanning
==============================


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


