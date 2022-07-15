#!/bin/bash -l 
usage(){ cat << EOU
OPTICKS_INPUT_PHOTON.sh Configuring SEventConfig::InputPhoton default
=======================================================================

EOU
}

#path=RandomSpherical10_f8.npy
vers=down
#path=/tmp/storch_test/$vers/$(uname)/ph.npy
path=storch_test/$vers/$(uname)/ph.npy

if [ -n "$path" ]; then 
    OPTICKS_INPUT_PHOTON=$path
   
    if [ "${path:0:1}" == "/" -o "${path:0:1}" == "$" ]; then 
        abspath=$path
    else
        abspath=$HOME/.opticks/InputPhotons/$path
    fi

    if [ -z "$QUIET" ]; then 
        if [ ! -f "$abspath" ]; then 
            echo == $BASH_SOURCE : path $path abspath $abspath DOES NOT EXIST 
            echo == $BASH_SOURCE : create with ana/input_photons.sh OR sysrap/tests/storch_test.sh 
        fi 
    fi 

    OPTICKS_INPUT_PHOTON_ABSPATH=$abspath

    case ${GEOM:-Dummy} in 
       J0*) OPTICKS_INPUT_PHOTON_FRAME=Hama:0:1000 ;;
    esac

fi 


