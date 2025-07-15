#!/bin/bash
usage(){ cat << EOU
U4VolumeMakerTest.sh
=====================

u4t::

   ./U4VolumeMakerTest.sh

HMM when the GEOM has manager prefix hama/nnvt/hmsk/nmsk
U4VolumeMaker::Make currently assumes the rest of the name
is of an LV.

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

bin=U4VolumeMakerTest

arg=${1:-run}

#geom=BoxOfScintillator  # default
#geom=hama_body_log
#geom=hamaLogicalPMT
#geom=hamaBodyLog
#geom=V1J008
#geom=V1J011
geom=BigWaterPool

export GEOM=${GEOM:-$geom}
#source $HOME/.opticks/GEOM/GEOM.sh

origin=$HOME/.opticks/GEOM/$GEOM/origin.gdml
if [ -f "$origin" ]; then
   export ${GEOM}_GDMLPath=$origin
   export U4VolumeMaker=INFO
fi


vv="BASH_SOURCE arg geom GEOM bin"

if [ "${arg/info}" != "${arg}" ]; then
    for v in $vv ; do printf "%20s : %s \n" "$v" "${!v}" ; done
fi

if [ "${arg/run}" != "${arg}" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1
fi

if [ "${arg/dbg}" != "${arg}" ]; then
    source dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi

exit 0
