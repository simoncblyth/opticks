#!/bin/bash -l 

export CFBASE=$HOME/.opticks/ntds3/G4CXOpticks
export FOLD=$CFBASE/G4CXSimulateTest/ALL

${IPYTHON:-ipython} --pdb -i check.py 

