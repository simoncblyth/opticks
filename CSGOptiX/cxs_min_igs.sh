#!/bin/bash -l 
usage(){ cat << EOU
cxs_min_igs.sh
===============

::

   ~/o/CSGOptiX/cxs_min_igs.sh 

EOU
}


#export OPTICKS_NUM_EVENT=3   # reduce from default of 1000 for shakedown
#export OPTICKS_EVENT_MODE=DebugLite 

SDIR=$(dirname $(realpath $BASH_SOURCE))
#TEST=input_genstep LIFECYCLE=1 $SDIR/cxs_min.sh 
TEST=input_genstep $SDIR/cxs_min.sh 
