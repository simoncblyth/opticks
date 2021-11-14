#!/bin/bash -l 
usage(){ cat << EOU
xxs.sh : Geant4 equivalent to OptiX cxs.sh 
===============================================

Provides 2D cross section plots of G4VSolid provided from j/PMTSim 

0.0             -2.5            -5.0            -179.2          -242.5          -275.0          -385.0          -420.0  zdelta  
                                                                                                                                
190.0           0.0             -5.0            -162.0          -210.0          -275.0          -350.0          -420.0  az1     
0.0             -5.0            -195.0          -210.0          -275.0          -365.0          -420.0          -450.0  az0     

   190,0,-5,-195,-162,-210,-275,-365,-350,-420,-450



EOU
}

msg="=== $BASH_SOURCE :"

export JUNO_PMT20INCH_POLYCONE_NECK=ENABLED
export JUNO_PMT20INCH_SIMPLIFY_CSG=ENABLED
#export JUNO_PMT20INCH_NOT_USE_REAL_SURFACE=ENABLED
export JUNO_PMT20INCH_PLUS_DYNODE=ENABLED    ## xxs restricted to single solids so this not so useful

#geom=orb
#geom=UnionOfHemiEllipsoids
#geom=UnionOfHemiEllipsoids-50
#geom=pmt_solid
#geom=I
#geom=III
#geom=1_2
#geom=1_3

#geom=_pmt_cut_solid
#geom=pmt_solid
geom=body_solid
#geom=body_solid_zcut
#geom=body_solid_zcut,body_solid
#geom=inner2_solid_zcut
#geom=pmt_solid_zcut
#geom=body_solid,inner2_solid   

#geom=CutTail_HamaPMT_Solid,inner2_solid

export GEOM=${GEOM:-$geom}

tmp=/tmp/$USER/opticks
reldir=extg4/X4IntersectTest
fold=$tmp/$reldir 

echo $msg reldir $reldir fold $fold 

#other_reldir=GeoChain
other_reldir=CSGOptiX/CSGOptiXSimulateTest
other_fold=$tmp/$other_reldir

#if [ ! -d "$fold" ]; then
#    echo $msg reldir $reldir fold $fold MUST EXIST 
#    exit 1 
#fi

if [ -d "$other_fold" ]; then
    other_exists=YES
else
    other_exists=NO
fi  
echo $msg other_reldir $other_reldir other_fold $other_fold other_exists $other_exists


if [ "$GEOM" == "orb" ]; then

    dz=0
    num_pho=10
    cegs=16:0:9:0:0:$dz:$num_pho
    gridscale=0.15

    zz=-100,100
    xx=-100,100

else

    dz=-4
    num_pho=10
    #cegs=16:0:9:0:0:$dz:$num_pho
    #gridscale=0.15
    cegs=9:0:16:0:0:$dz:$num_pho
    gridscale=0.10

    #zz=190,0,-5,-162,-195,-210,-275,-350,-365,-400,-420,-450
    zz=190,-162,-195,-210,-275,-350,-365,-420,-450
    zzd=-183.2246  
    xx=-254,254

    unset CXS_OVERRIDE_CE
    export CXS_OVERRIDE_CE=0:0:-130:320   ## fix at the full uncut ce 
    # -320-130 = -450  320-130 = 190 
fi 





export GRIDSCALE=${GRIDSCALE:-$gridscale}
export CXS_CEGS=${CXS_CEGS:-$cegs}
export CXS_RELDIR=${CXS_RELDIR:-$reldir} 
export CXS_OTHER_RELDIR=${CXS_OTHER_RELDIR:-$other_reldir} 

env | grep CXS

export XX=${XX:-$xx}
export ZZ=${ZZ:-$zz}
export ZZD=${ZZD:-$zzd}

arg=${1:-runana}

if [ "${arg/exit}" != "$arg" ]; then
   echo $msg early exit 
   exit 0 
fi

if [ "${arg/run}" != "$arg" ]; then
    $GDB X4IntersectTest
    [ $? -ne 0 ] && echo run error && exit 1 
fi  

if [ "${arg/ana}"  != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i tests/X4IntersectTest.py 
    [ $? -ne 0 ] && echo ana error && exit 2
fi 

exit 0 
