#!/bin/bash
usage(){ cat << EOU

~/o/CSG/tests/CSGSimtraceTest.sh


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=CSGSimtraceTest

defarg="info_run"
arg=${1:-$defarg}

source $HOME/.opticks/GEOM/GEOM.sh

vv="BASH_SOURCE defarg arg PWD GEOM"


if [ "${arg/info}" != "$arg" ]; then
   for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
   $name
   [ $? -ne 0 ] && echo $BASH_SOURCE  - run error && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
   source dbg__.sh
   dbg__ $name
   [ $? -ne 0 ] && echo $BASH_SOURCE  - dbg error && exit 2
fi





exit 0

