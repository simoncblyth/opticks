#!/bin/bash -l 
usage(){ cat << EOU
OPTICKS_INPUT_PHOTON.sh Configuring SEventConfig::InputPhoton default
=======================================================================

This script sets the OPTICKS_INPUT_PHOTON envvar, it is sourced by:

* u4/tests/U4RecorderTest.sh 
* gx/gxs.sh   

The envvar value is used as the default of SEventConfig::InputPhoton

EOU
}

#path=RandomSpherical10_f8.npy
path=/tmp/storch_test/out/$(uname)/ph.npy
export OPTICKS_INPUT_PHOTON=$path

echo === $BASH_SOURCE :  OPTICKS_INPUT_PHOTON $OPTICKS_INPUT_PHOTON

