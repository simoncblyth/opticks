#!/bin/bash -l


dz=0
num_pho=10
cegs=16:0:9:0:0:$dz:$num_pho
gridscale=0.15

#reldir=extg4/X4IntersectTest
#cxs=orb

cxs=PMTSim_inner_solid_1_9
reldir=GeoChainSolidTest


export GRIDSCALE=${GRIDSCALE:-$gridscale}
export CXS_CEGS=${CXS_CEGS:-$cegs}
export CXS_RELDIR=${CXS_RELDIR:-$reldir} 
export CXS=${CXS:-$cxs}

arg=${1:-ana}

if [ "$arg" == "run" ]; then
    $GDB X4IntersectTest 
elif [ "$arg" == "ana" ]; then 
    ${IPYTHON:-ipython} --pdb -i tests/X4IntersectTest.py 
fi 




