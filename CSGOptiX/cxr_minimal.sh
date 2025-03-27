#!/bin/bash
usage(){ cat << EOU
cxr_minimal.sh
===============

Does minimal env setup to visualize the last persisted geometry 
using CSGOptiXRenderInteractiveTest.

See also cxr_min.sh which does similar but is a lot less minimal.

EOU
}

geombase=$HOME/.opticks/GEOM
last_CSGFoundry=$(cd $geombase && ls -1dt */CSGFoundry | head -1)
last_GEOM=$(dirname $last_CSGFoundry)

export GEOM=$last_GEOM
export ${GEOM}_CFBaseFromGEOM=$geombase/$GEOM

bin=CSGOptiXRenderInteractiveTest
tbin=$(which $bin)

vv="0 geombase last_CSGFoundry last_GEOM GEOM ${GEOM}_CFBaseFromGEOM vv bin tbin"
vvp(){ for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done ; }

vvp 
$bin
vvp

