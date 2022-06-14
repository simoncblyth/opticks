#!/bin/bash -l 

usage(){ cat << EOU
input_photons.sh
===================

Running the script creates a selection if input photon .npy arrays
into  ~/.opticks/InputPhotons

Change the DTYPE envvar to switch beween precisions np.float32 OR np.float64

::

   cd ~/opticks/ana
   ./input_photons.sh 

EOU
}

export DTYPE=np.float32
#export DTYPE=np.float64

${IPYTHON:-ipython} --pdb input_photons.py -- $*

ls -alst ~/.opticks/InputPhotons

