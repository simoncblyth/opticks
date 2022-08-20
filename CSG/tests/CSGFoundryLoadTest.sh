#!/bin/bash -l 

usage(){ cat << EOU



::

    export CFBASE=$HOME/.opticks/ntds3/G4CXOpticks
    if [ ! -f "$CFBASE/CSGFoundry/solid.npy" ]; then
       echo $BASH_SOURCE : ERROR : CFBASE dir does not contain CSGFoundry/solid.npy
       exit 1
    fi  
    CSGFoundryLoadTest

EOU
}

source $OPTICKS_HOME/bin/GEOM_.sh 
CSGFoundryLoadTest



