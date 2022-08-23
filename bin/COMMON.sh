#!/bin/bash -l 
usage(){ cat << EOU
COMMON.sh 
============

Shortcut bash functions:

com_
    Edit the COMMON.sh config scripts
geom_
    Edit the GEOM_.sh config
oip
    Edit the OPTICKS_INPUT_PHOTON.sh config 

EOU
}


bindir=$(dirname $BASH_SOURCE)

source $bindir/GEOM_.sh                   # defines and exports : GEOM, GEOMDIR 
source $bindir/OPTICKS_INPUT_PHOTON.sh    # defines and exports : OPTICKS_INPUT_PHOTON


upfind_cfbase(){
    : COMMON.sh : traverse directory tree upwards searching CFBase geometry dir identified by existance of CSGFoundry/solid.npy  
    local dir=$1
    while [ ${#dir} -gt 1 -a ! -f "$dir/CSGFoundry/solid.npy" ] ; do dir=$(dirname $dir) ; done 
    echo $dir
}


if [ "$GEOM" == "J000" -o "$GEOM" == "J001" -o "$GEOM" == "J002" -o "$GEOM" == "J003" ]; then 

   case $GEOM in 
     J000) OPTICKS_INPUT_PHOTON_FRAME=NNVT:0:1000 ;;
     J001) OPTICKS_INPUT_PHOTON_FRAME=Hama:0:1000 ;;
     J002) OPTICKS_INPUT_PHOTON_FRAME=NNVT:0:1000 ;;
     J003) OPTICKS_INPUT_PHOTON_FRAME=NNVT:0:1000 ;;
   esac

   [ -n "$OPTICKS_INPUT_PHOTON_FRAME" ] && export OPTICKS_INPUT_PHOTON_FRAME
   [ -n "$OPTICKS_INPUT_PHOTON_FRAME" ] && export MOI=$OPTICKS_INPUT_PHOTON_FRAME

   export U4VolumeMaker_PVG_WriteNames=1
   export U4VolumeMaker_PVG_WriteNames_Sub=1
fi 

if [ -z "$QUIET" ]; then 
    vars="BASH_SOURCE GEOM OPTICKS_INPUT_PHOTON OPTICKS_INPUT_PHOTON_FRAME MOI"
    for var in $vars ; do printf "%30s : %s\n" $var ${!var} ; done 
fi 

