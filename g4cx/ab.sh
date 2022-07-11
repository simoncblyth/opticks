#!/bin/bash -l 

QUIET=1 gx
A_FOLD=$(./gxs.sh fold)

QUIET=1 u4
B_FOLD=$(./u4s.sh fold)

gx
source ../bin/AB_FOLD.sh 

export A_FOLD
export B_FOLD

${IPYTHON:-ipython} --pdb -i tests/G4CXSimulateTest_ab.py




