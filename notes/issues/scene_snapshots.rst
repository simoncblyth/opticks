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


