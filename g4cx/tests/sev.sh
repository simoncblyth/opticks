#!/bin/bash -l 
usage(){ cat << EOU
sev.sh
=========

Update geometry, transfer to laptop, load into ipython::

    ntds3       # on workstation
    ntds3       # on mac laptop : grabs persisted geometry from workstation
    gxt         # cd ~/opticks/g4cx/tests
    ./sev.sh    # load into ipython

Alternative jump into ipython from anywhere::

    cd
    ~/opticks/g4cx/tests/sev.sh 

EOU
}

export BASE=/tmp/$USER/opticks/ntds3/G4CXOpticks

export CFBASE=$BASE
export STBASE=$BASE

#export FOLD=$CFBASE/stree

export FOLD=$(dirname $CFBASE)/ALL/z000
#export FOLD=$(dirname $CFBASE)/ALL/p001


${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/sev.py 


