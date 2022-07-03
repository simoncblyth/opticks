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

if [ -n "$path" ]; then 
    export OPTICKS_INPUT_PHOTON=$path

    if [ "${path:0:1}" == "/" -o "${path:0:1}" == "$" ]; then 
        abspath=$path
    else
        abspath=$HOME/.opticks/InputPhotons/$path
    fi
    if [ ! -f "$abspath" ]; then 
        echo $msg path $path abspath $abspath DOES NOT EXIST : create with ana/input_photons.sh OR sysrap/tests/storch_test.sh 
        exit 1 
    else
        echo $msg path $path abspath $abspath exists 
    fi 
fi 

echo === $BASH_SOURCE : OPTICKS_INPUT_PHOTON $OPTICKS_INPUT_PHOTON
