#!/bin/bash
usage(){ cat << EOU

::
  
   ~/o/g4cx/tests/G4CXTest_raindrop_simtrace.sh 


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
B_SIMTRACE=1 ./G4CXTest_raindrop.sh $*


