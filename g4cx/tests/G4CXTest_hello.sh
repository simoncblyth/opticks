#!/bin/bash
usage(){ cat << EOU
G4CXTest_hello.sh : Standalone bi-simulation with G4CXApp::Main
===================================================================

Notice the GEOM and _GDMLPath envvars they get interpreted
by U4VolumeMaker::PV (specifically U4VolumeMaker::PVG_) to
specify loading geometry from GDML.

EOU
}

cd $(dirname $BASH_SOURCE)

bin=G4CXTest

#gdml=/path/to/some.gdml
gdml=$HOME/.opticks/GEOM/V1J011/origin.gdml

export GEOM=hello             # GEOM is identifier for a geometry
export hello_GDMLPath=$gdml   # associate a GDMLPath with that geometry

if [ ! -f "$gdml" ]; then
   echo $BASH_SOURCE : error : NO GDML AT $gdml
   exit 1
fi

defarg="info_run"
arg=${1:-$defarg}

vars="BASH_SOURCE GEOM ${GEOM}_GDMLPath"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
    source dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi


exit 0

