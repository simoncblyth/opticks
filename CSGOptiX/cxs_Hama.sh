#!/bin/bash -l 

#geom=Hama_1
geom=HamaXZ_1
#geom=HamaYZ_1
#geom=HamaXY_1

#geom=Hama_2
#geom=Hama_4
#geom=Hama_8
#geom=Hama_16

export GEOM=${GEOM:-$geom}

isel=
cfbase=
ce_offset=0
ce_scale=0
gsplot=1


if [ "$GEOM" == "Hama_1" ]; then

    moi=Hama
    cegs=16:0:9:500   # XZ works 
    gridscale=0.10

elif [ "$GEOM" == "HamaXZ_1" ]; then

    moi=Hama
    cegs=16:0:9:500   
    gridscale=0.10

elif [ "$GEOM" == "HamaYZ_1" ]; then

    moi=Hama
    cegs=0:16:9:500  
    gridscale=0.10

elif [ "$GEOM" == "HamaXY_1" ]; then

    moi=Hama
    cegs=16:9:0:500 
    gridscale=0.10

elif [ "$GEOM" == "Hama_2" ]; then

    moi=Hama
    cegs=32:0:18:500
    gridscale=0.10

elif [ "$GEOM" == "Hama_4" ]; then

    moi=Hama
    cegs=64:0:36:100
    #gridscale=0.10
    gridscale=0.20
    gsplot=0

elif [ "$GEOM" == "Hama_8" ]; then

    moi=Hama
    cegs=128:0:72:100
    gridscale=0.40
    gsplot=0

elif [ "$GEOM" == "Hama_16" ]; then

    ##  CUDA error on synchronize with error 'an illegal memory access was encountered' (/data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:342)
    moi=Hama
    cegs=256:0:144:10
    gridscale=0.20
    gsplot=0

else

    echo $msg ERROR GEOM $GEOM unhandled 
    exit 1 

fi

source ./cxs.sh $*

