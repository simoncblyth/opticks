#!/bin/bash -l 

usage(){ cat << EOU
input_photons_plt.sh
=====================

::

   cd ~/opticks/ana
   ./input_photons.sh    # generate the input photons 
   ./input_photons_plt.sh 

EOU
}


#stem=RandomSpherical100
#stem=RandomDisc100
#stem=UpXZ1000
stem=DownXZ1000

path=${stem}_f4.npy

export OPTICKS_INPUT_PHOTON=${OPTICKS_INPUT_PHOTON:-$path}

${IPYTHON:-ipython} --pdb -i input_photons_plt.py -- $*




