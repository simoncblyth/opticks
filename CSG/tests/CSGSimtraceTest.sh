#!/bin/bash
usage(){ cat << EOU

~/o/CSG/tests/CSGSimtraceTest.sh


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=CSGSimtraceTest

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
LOGDIR=$TMP/$name
mkdir -p $LOGDIR

defarg="info_run_ls"
arg=${1:-$defarg}

source $HOME/.opticks/GEOM/GEOM.sh

vv="BASH_SOURCE defarg arg PWD GEOM LOGDIR"


if [ "${arg/info}" != "$arg" ]; then
   for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
   pushd $LOGDIR
   $name
   [ $? -ne 0 ] && echo $BASH_SOURCE  - run error && exit 1
   popd
fi

if [ "${arg/dbg}" != "$arg" ]; then
   pushd $LOGDIR
   source dbg__.sh
   dbg__ $name
   [ $? -ne 0 ] && echo $BASH_SOURCE  - dbg error && exit 2
   popd
fi

if [[ "$arg" =~ ls ]]; then
   echo ls -alst $LOGDIR
   ls -alst $LOGDIR
fi

exit 0

