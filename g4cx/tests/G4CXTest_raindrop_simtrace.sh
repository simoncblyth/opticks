#!/bin/bash
usage(){ cat << EOU
G4CXTest_raindrop_simtrace.sh
==============================

::

   ~/o/g4cx/tests/G4CXTest_raindrop_simtrace.sh


Note that definition of the world volume is grotty.
Probably Geant4 does not like intersecting from
origins outside world.

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

#export CEGS=16:0:9:2000
export CEGS=11:0:11:2000

defarg=run_tra
arg=${1:-$defarg}

B_SIMTRACE=1 ./G4CXTest_raindrop.sh $arg


