#!/bin/bash -l 

geom=uni_acrylic1_0
#geom=uni_acrylic3_0
#geom=uni_acrylic1_wide
#geom=uni_acrylic1_tight

export GEOM=${GEOM:-$geom}

isel=
cfbase=
ce_offset=0
ce_scale=0
gsplot=1


if [ "$GEOM" == "uni_acrylic1_0" ]; then
    moi=uni_acrylic1
    cegs=16:0:9:100
    gridscale=0.05

elif [ "$GEOM" == "uni_acrylic3_0" ]; then
    ## when use the option --additionacrylic-simplify-csg the uni_acrylic3 is not present 
    ## instead get uni_acrylic1 : is that OK? 
    moi=uni_acrylic3
    cegs=16:0:9:100
    #cegs=0:0:0:1000
    #cegs=16:4:9:100
    gridscale=0.05

elif [ "$GEOM" == "uni_acrylic3_wide" ]; then
    moi=uni_acrylic3
    cegs=32:0:18:100
    gridscale=0.025

elif [ "$GEOM" == "uni_acrylic3_tight" ]; then

    note="very tight grid to get into close corners"
    moi=uni_acrylic3
    cegs=16:0:9:100
    gridscale=0.025

else
    echo $msg ERROR GEOM $GEOM unhandled 
    exit 1 
fi

source ./cxs.sh 


