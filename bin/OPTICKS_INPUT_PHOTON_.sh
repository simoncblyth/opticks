#!/bin/bash -l 
usage(){ cat << EOU
OPTICKS_INPUT_PHOTON.sh Configuring SEventConfig::InputPhoton default
=======================================================================

Usage::

    source $OPTICKS_HOME/bin/OPTICKS_INPUT_PHOTON.sh

Sourcing this script defines, but does NOT export::

    OPTICKS_INPUT_PHOTON
    OPTICKS_INPUT_PHOTON_ABSPATH
    OPTICKS_INPUT_PHOTON_FRAME

The values depend on the GEOM variable. 

EOU
}

case ${GEOM:-Dummy} in 
    RaindropRockAirWater)      path=storch_test/down/$(uname)/ph.npy ;;
    RaindropRockAirWaterSD)    path=storch_test/down/$(uname)/ph.npy ;;
    RaindropRockAirWaterSmall) path=storch_test/up99/$(uname)/ph.npy ;;
                            *) path=RandomSpherical10_f8.npy  ;;
esac

if [ -n "$path" ]; then 
    OPTICKS_INPUT_PHOTON=$path
    if [ "${path:0:1}" == "/" -o "${path:0:1}" == "$" ]; then 
        abspath=$path
    else
        abspath=$HOME/.opticks/InputPhotons/$path
    fi
    OPTICKS_INPUT_PHOTON_ABSPATH=$abspath

    if [ -z "$QUIET" ]; then 
        if [ -n "$OPTICKS_INPUT_PHOTON_ABSPATH" -a ! -f "$OPTICKS_INPUT_PHOTON_ABSPATH"  ]; then 
            echo == $BASH_SOURCE : OPTICKS_INPUT_PHOTON_ABSPATH $OPTICKS_INPUT_PHOTON_ABSPATH  DOES NOT EXIST 
            echo == $BASH_SOURCE : create with ana/input_photons.sh OR sysrap/tests/storch_test.sh 
        fi 
    fi 
fi 

case ${GEOM:-Dummy} in 
    J0*) OPTICKS_INPUT_PHOTON_FRAME=Hama:0:1000 ;;
esac

if [ -z "$QUIET" ]; then 
    vars="BASH_SOURCE GEOM OPTICKS_INPUT_PHOTON OPTICKS_INPUT_PHOTON_FRAME OPTICKS_INPUT_PHOTON_ABSPATH"
    for var in $vars ; do printf "%30s : %s\n" $var ${!var} ; done 
fi 

