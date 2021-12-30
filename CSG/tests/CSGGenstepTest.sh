#!/bin/bash -l 

usage(){ cat << EOU
CSGGenstepTest.sh
===================

CSGGenstepTest checks the center-extent-gensteps used in CSGOptiX/cxs.sh 
by generating some photons on CPU from them and loading into python. 

EOU
}


#geom=HamaXZ_1
geom=XJfixtureConstruction_0

export GEOM=${GEOM:-$geom}
ce_offset=0

if [ "$GEOM" == "HamaXZ_1" ]; then
    moi=Hama
    cegs=16:0:9:500   
    gridscale=0.10
    ce_offset=0


elif [ "$GEOM" == "XJfixtureConstruction_0" ]; then

    moi="solidXJfixture:10"
    cegs=16:0:9:100               # XZ
    #cegs=0:16:9:100               # YZ
    gridscale=0.05
    ce_offset=1
fi 


export MOI=${MOI:-$moi}
export CXS_CEGS=${CXS_CEGS:-$cegs}
export CE_OFFSET=${CE_OFFSET:-$ce_offset}
export GRIDSCALE=${GRIDSCALE:-$gridscale}

CSGGenstepTest

${IPYTHON:-ipython} -i CSGGenstepTest.py 



