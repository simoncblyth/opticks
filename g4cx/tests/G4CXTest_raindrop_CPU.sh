#!/bin/bash -l 
usage(){ cat << EOU

::
  
   ~/o/g4cx/tests/G4CXTest_raindrop_CPU.sh 


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

OPTICKS_INTEGRATION_MODE=2 ./G4CXTest_raindrop.sh $*


