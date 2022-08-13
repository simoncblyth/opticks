#!/bin/bash -l 

usage(){ cat << EOU
iid.sh : comparing sensor info between the old world GGeo/GMergedMesh/GNodeLib and new world U4Tree/stree
===============================================================================================================


EOU
}


export BASE=/tmp/$USER/opticks/ntds3/G4CXOpticks

#export FOLD=$BASE/stree
export FOLD=$BASE/stree_reorderSensors


${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/iid.py 

