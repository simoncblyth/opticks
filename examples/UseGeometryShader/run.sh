#!/bin/bash -l 

DIR=$(dirname $(realpath $BASH_SOURCE))

#fold=/data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL/p003 
fold=/tmp/sphoton_test

export RECORD_FOLD=$fold
$DIR/build.sh $*




