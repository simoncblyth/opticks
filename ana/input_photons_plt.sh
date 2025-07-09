#!/bin/bash

usage(){ cat << EOU
input_photons_plt.sh
=====================

::

   ~/o/ana/input_photons.sh    # generate the input photons
   ~/o/ana/input_photons_plt.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

script=input_photons_plt.py


#stem=RandomSpherical100
#stem=RandomDisc100
#stem=UniformDisc_R500_10k
#stem=GridXY_X700_Z230_10k
#stem=GridXY_X1000_Z1000_40k
#stem=UpXZ1000
#stem=DownXZ1000
#stem=RainXZ1000
#stem=RainXZ_Z230_1000

#stem=CircleXZ_R500_100k
#stem=CircleXZ_R10_361
stem=SemiCircleXZ_R-500_100k

#sufx=_f4
sufx=_f8

STEM=${STEM:-$stem}


path=${STEM}${sufx}.npy

mode=3
export MODE=${MODE:-$mode}
export OPTICKS_INPUT_PHOTON=${OPTICKS_INPUT_PHOTON:-$path}


${IPYTHON:-ipython} --pdb -i $script -- $*




