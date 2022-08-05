#!/bin/bash -l 
usage(){ cat << EOU
ntds3.sh
=========

Update geometry, transfer to laptop, load into ipython::

    ntds3       # on workstation
    ntds3       # on mac laptop : grabs persisted geometry from workstation
    gxt         # cd ~/opticks/g4cx/tests
    ./ntds3.sh  # load into ipython

Alternative jump into ipython from anywhere::

    cd
    ~/opticks/g4cx/tests/ntds3.sh 

EOU
}

export CFBASE=/tmp/$USER/opticks/ntds3/G4CXOpticks
export FOLD=$CFBASE/stree

${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/gx.py 


