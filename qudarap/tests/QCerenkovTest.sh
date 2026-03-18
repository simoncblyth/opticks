#!/bin/bash
usage(){ cat << EOU

~/o/qudarap/tests/QCerenkovTest.sh info

CURRENTLY FAILS FROM bnd assert in QCerenkov::MakeInstance

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=QCerenkovTest

bin=$name
script=$name.py


tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
tmpdir=$TMP/$name
mkdir -p $tmpdir


defarg="info_run_pdb"
arg=${1:-$defarg}

vv="BASH_SOURCE name bin script tmp TMP tmpdir defarg arg"

if [ "${arg/info}" != "$arg" ]; then
   for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
   which $bin
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE ERROR from run && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
   source dbg__.sh
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE ERROR from dbg && exit 1
fi

if [ "${arg/pdb}" != "$arg" ]; then
   ${IPYTHON:-ipython} -i --pdb $script
   [ $? -ne 0 ] && echo $BASH_SOURCE ERROR from pdb && exit 2
fi

if [ "${arg/ana}" != "$arg" ]; then
   ${PYTHON:-python} $script
   [ $? -ne 0 ] && echo $BASH_SOURCE ERROR from ana && exit 3
fi

exit 0

