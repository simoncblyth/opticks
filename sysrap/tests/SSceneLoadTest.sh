#!/bin/bash
usage(){ cat << EOU
SSceneLoadTest.sh
===================

Loads persisted SScene and dumps the desc::

   ~/o/sysrap/tests/SSceneLoadTest.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SSceneLoadTest
bin=$name

source $HOME/.opticks/GEOM/GEOM.sh
source $HOME/.opticks/GEOM/ELV.sh
vars="BASH_SOURCE PWD GEOM ELV name bin"

defarg=info_run
arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

if [ "${arg/dbg}" != "$arg" ]; then
    source dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi

exit 0
