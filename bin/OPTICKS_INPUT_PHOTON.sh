#!/bin/bash -l 
usage(){ cat << EOU
OPTICKS_INPUT_PHOTON.sh Configuring SEventConfig::InputPhoton default
=======================================================================

This script sets envvars that are used as SEventConfig defaults. 

OPTICKS_INPUT_PHOTON 
   name or path of input photon .npy file, default of SEventConfig::InputPhoton 

OPTICKS_INPUT_PHOTON_FRAME
   moi_or_iidx string eg "Hama:0:1000" OR "35000", default of SEventConfig::InputPhotonFrame

This script is sourced by:

* u4/u4s.sh
* gx/gxs.sh   


EOU
}

#path=RandomSpherical10_f8.npy
path=/tmp/storch_test/out/$(uname)/ph.npy

if [ -n "$path" ]; then 
    export OPTICKS_INPUT_PHOTON=$path
    export OPTICKS_INPUT_PHOTON_FRAME=Hama:0:1000

    if [ "${path:0:1}" == "/" -o "${path:0:1}" == "$" ]; then 
        abspath=$path
    else
        abspath=$HOME/.opticks/InputPhotons/$path
    fi

    if [ -z "$QUIET" ]; then 
        if [ ! -f "$abspath" ]; then 
            echo == $BASH_SOURCE path $path abspath $abspath DOES NOT EXIST : create with ana/input_photons.sh OR sysrap/tests/storch_test.sh 
        else
            echo == $BASH_SOURCE path $path abspath $abspath exists 
        fi 
    fi 
fi 

if [ -z "$QUIET" ]; then 
    echo == $BASH_SOURCE : OPTICKS_INPUT_PHOTON $OPTICKS_INPUT_PHOTON
    echo == $BASH_SOURCE : OPTICKS_INPUT_PHOTON_FRAME $OPTICKS_INPUT_PHOTON_FRAME
fi
