#!/bin/bash -l 
usage(){ cat << EOU
cf.sh
======

Comparison of geometry python dumped from stree and CSGFoundry 

EOU
}


SDIR=$(dirname $BASH_SOURCE)

geom=J007
ridx=7

export GEOM=${GEOM:-$geom}
export RIDX=${RIDX:-$ridx}

#$SDIR/sysrap/tests/stree_load_test.sh ana
$SDIR/CSG/tests/CSGFoundryLoadTest.sh ana


