#!/bin/bash -l 

export FOLD=/tmp/$USER/opticks/U4TreeTest

${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/sfreq.py 

