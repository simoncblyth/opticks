#!/bin/bash 
usage(){ cat << EOU
AB_FOLD.sh
============

This script is expected to be sourced, it exports envvars A_FOLD and B_FOLD
which are used for comparing output arrays for example by:

1. u4/tests/U4RecorderTest_ab.sh 
2. sysrap/dv.sh 


EOU
}

export A_FOLD=/tmp/$USER/opticks/GeoChain/BoxedSphere/CXRaindropTest
export B_FOLD=/tmp/$USER/opticks/U4RecorderTest

vars="BASH_SOURCE A_FOLD B_FOLD"
for var in $vars ; do printf "%25s : %s \n" "$var" "${!var}" ; done 




