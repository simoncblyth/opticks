#!/bin/bash -l 
usage(){ cat << EOU
cxr_min.sh
===========

Shakedown cxr.sh scripts using this minimal approach. 

EOU
}

bin=CSGOptiXRenderTest

geom=V0J008
export GEOM=${GEOM:-$geom}
export ${GEOM}_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM


$bin





