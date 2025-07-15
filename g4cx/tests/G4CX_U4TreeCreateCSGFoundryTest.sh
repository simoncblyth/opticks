#!/bin/bash
usage(){ cat << EOU
G4CX_U4TreeCreateCSGFoundryTest.sh
===================================

Creates Geant4 PV configured with GEOM envvar,
converts to Opticks stree/CSGFoundry and persists
the CSGFoundry into ~/.opticks/GEOM.

Visualize the result with ~/o/cxr_min.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=G4CX_U4TreeCreateCSGFoundryTest

source $HOME/.opticks/GEOM/GEOM.sh
fold=$HOME/.opticks/GEOM/$GEOM
export FOLD=$fold

defarg="info_run"
arg=${1:-$defarg}

vv="BASH_SOURCE name GEOM fold FOLD defarg arg"

if [ "${arg/info}" != "$arg" ]; then
   for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then

    if [ -d "$FOLD" ]; then
       ans="NO"
       echo $BASH_SOURCE FOLD $FOLD exists already
       read -p "$BASH_SOURCE - Enter YES to proceed to overwrite into this FOLD : " ans
    else
       ans="YES"
    fi

    if [ "$ans" == "YES" ]; then
       $name
       [ $? -ne 0 ] && echo $BASH_SOURCE - run error && exit 1
    else
       echo $BASH_SOURCE - skipping
    fi
fi


exit 0


