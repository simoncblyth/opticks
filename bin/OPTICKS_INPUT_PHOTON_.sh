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
                    J000)      label=DownXZ1000    ;; 
           hama_body_log)      label=DownXZ1000    ;; 
    RaindropRockAirWater)      label=storchdown ;;
    RaindropRockAirWaterSD)    label=storchdown  ;;
    RaindropRockAirWaterSmall) label=storchM1up99 ;;
   # RaindropRockAirWaterSmall) label=UpXZ1000    ;; 
                            *) label=RandomSpherical10  ;;
esac

case ${GEOM:-Dummy} in 
    J000) OPTICKS_INPUT_PHOTON_FRAME=Hama:0:1000 ;;
esac

case ${label:-dummy} in 
          storchdown) path=storch_test/down/$(uname)/ph.npy ;;   ## TODO: remove 
        storchM1up99) path=storch/M1up99/ph.npy ;; 
                   *) path=${label}_f8.npy ;; 
esac


if [ -n "$path" ]; then 
    OPTICKS_INPUT_PHOTON_LABEL=$label  
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


if [ -z "$QUIET" ]; then 
    vars="BASH_SOURCE GEOM OPTICKS_INPUT_PHOTON OPTICKS_INPUT_PHOTON_FRAME OPTICKS_INPUT_PHOTON_ABSPATH OPTICKS_INPUT_PHOTON_LABEL"
    for var in $vars ; do printf "%30s : %s\n" $var ${!var} ; done 
fi 

