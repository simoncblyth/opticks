#!/bin/bash -l 

usage(){ cat << EOU
xxv.sh : Volume equivalent of xxs.sh using extg4/tests/X4IntersectVolumeTest.cc
==================================================================================

Provides 2D cross section plots of all the G4VSolid from a PV tree of solids 
with structure transforms applied to intersects.

To run over all the commented and uncommented geom listed in xxv.sh below use ./xxv_scan.sh 


TODO: improve presentation, currently too small : perhaps update to use sframe.cc/py 

EOU
}

msg="=== $BASH_SOURCE :"
bin=X4IntersectVolumeTest
reldir=extg4/$bin


loglevels(){
    export X4Intersect=INFO
    export SCenterExtentGenstep=INFO
    export SFrameGenstep=INFO
    export PMTSim=2 

}
loglevels


# HMM: should be using GEOM_.sh ? 

#geom=hamaBodyPhys:nurs
#geom=hamaBodyPhys:nurs:pdyn
#geom=hamaBodyPhys:nurs:pdyn:prtc:obto
#geom=hamaBodyPhys:pdyn
#geom=hamaBodyPhys

geom=hamaLogicalPMTWrapLV

#geom=nnvtBodyPhys:nurs
#geom=nnvtBodyPhys:nurs:pdyn
#geom=nnvtBodyPhys:nurs:pdyn:prtc:obto
#geom=nnvtBodyPhys:pdyn
#geom=nnvtBodyPhys


#export GEOM=${GEOM:-$geom}
export X4IntersectVolumeTest_GEOM=${GEOM:-$geom}
export FOLD=/tmp/$USER/opticks/extg4/X4IntersectVolumeTest/$X4IntersectVolumeTest_GEOM

zcut=${geom#*zcut}
[ "$geom" != "$zcut" ] && zzd=$zcut 
echo geom $geom zcut $geom zzd $zzd


manual_config(){
    dz=-4
    num_pho=10
    cegs=9:0:16:0:0:$dz:$num_pho
    gridscale=0.10

    #zz=190,-162,-195,-210,-275,-350,-365,-420,-450
    xx=-254,254

    unset CXS_OVERRIDE_CE
    export CXS_OVERRIDE_CE=0:0:-130:320   ## fix at the full uncut ce 

    export GRIDSCALE=${GRIDSCALE:-$gridscale}
    export CXS_CEGS=${CXS_CEGS:-$cegs}
    export CXS_RELDIR=${CXS_RELDIR:-$reldir} 
    export CXS_OTHER_RELDIR=${CXS_OTHER_RELDIR:-$other_reldir} 

    env | grep CXS
}
#manual_config

export GRIDSCALE=0.08  ## HUH: why is this needed ? default scale wrong  ?


# presentational only 
export XX=${XX:-$xx}
export ZZ=${ZZ:-$zz}
export ZZD=${ZZD:-$zzd}





arg=${1:-runana}

if [ "${arg/exit}" != "$arg" ]; then
   echo $msg early exit 
   exit 0 
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi  
if [ "${arg/dbg}" != "$arg" ]; then

    case $(uname) in 
       Darwin) lldb__ $bin ;;
        Linux)  gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi  
if [ "${arg/ana}"  != "$arg" ]; then 

    if [ -n "$SCANNER" ]; then 
        ${IPYTHON:-ipython} --pdb  tests/$bin.py 
        [ $? -ne 0 ] && echo $BASH_SOURCE ana noninteractive error && exit 3
    else
        ${IPYTHON:-ipython} --pdb -i tests/$bin.py 
        [ $? -ne 0 ] && echo $BASH_SOURCE ana interactive error && exit 4
    fi
fi 


if [ "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    export CAP_BASE=$FOLD/figs
    export CAP_REL=xxv
    export CAP_STEM=$X4IntersectVolumeTest_GEOM
    case $arg in  
       mpcap) source mpcap.sh cap  ;;  
       mppub) source mpcap.sh env  ;;  
    esac

    if [ "$arg" == "mppub" ]; then 
        source epub.sh 
    fi  
fi 




exit 0 

