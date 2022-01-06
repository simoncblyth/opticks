#!/bin/bash -l 

if [ "$GEOM" == "25" ]; then
    cfbase=$TMP/CSGDemoTest/dcyl    
    moi=0
    cegs=16:0:9:100
    gridscale=0.025
    isel=0                           # setting isel to zero, prevents skipping bnd 0 
elif [ "$GEOM" == "30" ]; then
    note="HMM : box minus sub-sub cylinder NOT showing the spurious intersects, maybe nice round demo numbers effect"
    cfbase=$TMP/CSGDemoTest/bssc    
    moi=0
    cegs=16:0:9:100
    gridscale=0.025
    isel=0                           # setting isel to zero, prevents skipping bnd 0 
elif [ "$GEOM" == "100" ]; then
    cfbase=$TMP/GeoChain/AdditionAcrylicConstruction  
    moi=0
    cegs=16:0:9:100
    gridscale=0.1
    isel=0
elif [ "$GEOM" == "101" ]; then
    cfbase=$TMP/GeoChain/BoxMinusTubs1
    moi=0
    cegs=16:0:9:100
    gridscale=0.1
    isel=0

elif [ "$GEOM" == "SphereWithPhiSegment" ]; then
    cfbase=$TMP/GeoChain/$GEOM
    num_pho=100
    cegs=9:16:0:0:0:$dz:$num_pho
    gridscale=0.10

else
    # everything else assume single PMT dimensions
    cfbase=$TMP/GeoChain/$GEOM
    moi=0
    dz=-4
    num_pho=100
    #cegs=16:0:9:0:0:$dz:$num_pho
    cegs=9:0:16:0:0:$dz:$num_pho
    #gridscale=0.15
    gridscale=0.10
    isel=
    unset CXS_OVERRIDE_CE
    export CXS_OVERRIDE_CE=0:0:-130:320   ## fix at the full uncut ce 
fi 



source ./cxs.sh 


