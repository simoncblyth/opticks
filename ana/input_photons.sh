#!/bin/bash -l 
usage(){ cat << EOU
input_photons.sh
===================

Running this script creates several input photon .npy arrays 
into the directory ~/.opticks/InputPhotons::

   o   # cd ~/opticks

   ./ana/input_photons.sh   # create input photon arrays (non interactive)
   ./ana/iinput_photons.sh  # create input photon arrays (interactive) 

Note that the python script runs twice with DTYPE envvar 
as np.float32 and np.float64 in order to create the arrays 
in both single and double precision. 

EOU
}

DIR=$(dirname $BASH_SOURCE)

dtypes="np.float32 np.float64"
for dtype in $dtypes ; do 
    DTYPE=$dtype ${IPYTHON:-ipython} --pdb $OPT $DIR/input_photons.py -- $*
done

ls -alst ~/.opticks/InputPhotons

