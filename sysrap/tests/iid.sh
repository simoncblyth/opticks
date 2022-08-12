#!/bin/bash -l 

export BASE=/tmp/$USER/opticks/ntds3/G4CXOpticks
export FOLD=$BASE/stree

${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/iid.py 

