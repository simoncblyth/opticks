#!/bin/bash -l
usage(){ cat << EOU
xxs.sh : Geant4 equivalent to OptiX cxs.sh 
===============================================

Provides 2D cross section plots of G4VSolid provided from j/PMTSim 

EOU
}


export JUNO_PMT20INCH_POLYCONE_NECK=ENABLED
export JUNO_PMT20INCH_SIMPLIFY_CSG=ENABLED
export JUNO_PMT20INCH_NOT_USE_REAL_SURFACE=ENABLED


#cxs=orb
#cxs=pmt_solid
#cxs=I
#cxs=III
#cxs=1_2
cxs=1_3

export CXS=${CXS:-$cxs}
reldir=extg4/X4IntersectTest
#reldir=GeoChain
fold=/tmp/$USER/opticks/$reldir 

if [ ! -d "$fold" ]; then
    echo reldir $reldir fold $fold MUST EXIST 
    exit 1 
fi 

if [ "$CXS" == "orb" ]; then

    dz=0
    num_pho=10
    cegs=16:0:9:0:0:$dz:$num_pho
    gridscale=0.15

    zz=-100,100
    xx=-100,100

else

    dz=-4
    num_pho=10
    cegs=16:0:9:0:0:$dz:$num_pho
    gridscale=0.15

    zz=190,0,-5,-162,-195,-210,-275,-350,-365,-400,-420,-450
    xx=-254,254

    unset CXS_OVERRIDE_CE
    export CXS_OVERRIDE_CE=0:0:-130:320   ## fix at the full uncut ce 
    # -320-130 = -450  320-130 = 190 

fi 


export GRIDSCALE=${GRIDSCALE:-$gridscale}
export CXS_CEGS=${CXS_CEGS:-$cegs}
export CXS_RELDIR=${CXS_RELDIR:-$reldir} 
export XX=${XX:-$xx}
export ZZ=${ZZ:-$zz}

arg=${1:-runana}

if [ "${arg/run}" != "$arg" ]; then
    $GDB X4IntersectTest
    [ $? -ne 0 ] && echo run error && exit 1 
fi  

if [ "${arg/ana}"  != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i tests/X4IntersectTest.py 
    [ $? -ne 0 ] && echo ana error && exit 2
fi 

exit 0 
