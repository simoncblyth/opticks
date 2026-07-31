#!/bin/bash
usage(){ cat << EOU
G4CXTest_raindrop_animation.sh
================================

TODO: revive record animation


alt-A
   enable photon record aimimation rendering of AFOLD/record.npy
alt-B
   enable photon record aimimation rendering of BFOLD/record.npy



EOU
}


source /data1/blyth/local/opticks_Debug/envset.sh

export MOI=0

export AFOLD=/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/A000
export BFOLD=/data1/blyth/tmp/GEOM/RaindropRockAirWater/G4CXTest/ALL0_Debug_Philox/B000

export GEOM=RaindropRockAirWater
export _CFB=/home/blyth/.opticks/GEOM/$GEOM
export ${GEOM}_CFBaseFromGEOM=/home/blyth/.opticks/GEOM/$GEOM

export T0=0
export T1=4
export TT=0
export TN=1000


defarg=info_ls_render
arg=${1:-$defarg}


vv="BASH_SOURCE GEOM ${GEOM}_CFBaseFromGEOM _CFB AFOLD BFOLD"
ff="${GEOM}_CFBaseFromGEOM AFOLD BFOLD"

if [[ "$arg" =~ help ]]; then
    usage
fi

if [[ "$arg" =~ info ]]; then
    for v in $vv ; do printf "%50s : %s\n" "$v" "${!v}" ; done
fi

if [[ "$arg" =~ ls ]]; then
    for f in $ff ; do printf "ls -alst %s\n" "$f" && ls -alst ${!f} ; done
fi

if [[ "$arg" =~ render ]]; then
    cxr_min.sh
    [ $? -ne 0 ] && echo $BASH_SOURCE - ERROR from render && exit 1
fi

exit 0

