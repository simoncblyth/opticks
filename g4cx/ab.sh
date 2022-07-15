#!/bin/bash -l 

A_FOLD=$($OPTICKS_HOME/g4cx/gxs.sh fold)
B_FOLD=$($OPTICKS_HOME/u4/u4s.sh fold)

source $OPTICKS_HOME/bin/AB_FOLD.sh 

export A_FOLD
export B_FOLD

${IPYTHON:-ipython} --pdb -i $OPTICKS_HOME/g4cx/tests/G4CXSimulateTest_ab.py




