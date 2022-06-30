#!/bin/bash 
usage(){ cat << EOU
AB_FOLD.sh
============

This script is expected to be sourced, it exports envvars A_FOLD and B_FOLD
which are used for comparing output arrays for example by:

1. u4/tests/U4RecorderTest_ab.sh 
2. sysrap/dv.sh 

The A_FOLD and B_FOLD variables exported are sensitive to the FOLD_MODE
control envvar.   

FOLD_MODE:TMP
    used for everyday smaller samples that can easily be recreated, 
    for which being lost on rebooting the machine is not a problem

FOLD_MODE:KEEP
    used for larger samples which prefer not to loose at every reboot

EOU
}

export FOLD_MODE=${FOLD_MODE:-TMP}

A_FOLD_TMP=/tmp/$USER/opticks/GeoChain/BoxedSphere/CXRaindropTest
B_FOLD_TMP=/tmp/$USER/opticks/U4RecorderTest

A_FOLD_KEEP=/usr/local/opticks/tests/GeoChain/BoxedSphere/CXRaindropTest
B_FOLD_KEEP=/usr/local/opticks/tests/U4RecorderTest

A_FOLD_LOGF=/tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT
B_FOLD_LOGF=/tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_ORIGINAL_ShimG4OpRayleigh_ORIGINAL


case $FOLD_MODE in
  TMP)  export A_FOLD=$A_FOLD_TMP  ; export B_FOLD=$B_FOLD_TMP  ;; 
  KEEP) export A_FOLD=$A_FOLD_KEEP ; export B_FOLD=$B_FOLD_KEEP ;; 
  LOGF) export A_FOLD=$A_FOLD_LOGF ; export B_FOLD=$B_FOLD_LOGF ;; 
esac

vars="BASH_SOURCE FOLD_MODE A_FOLD B_FOLD"
for var in $vars ; do printf "%25s : %s \n" "$var" "${!var}" ; done 

echo A_FOLD $A_FOLD
du -hs $A_FOLD
du -hs $A_FOLD/*.npy

echo B_FOLD $B_FOLD
du -hs $B_FOLD
du -hs $B_FOLD/*.npy

