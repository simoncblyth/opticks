#!/bin/bash
usage(){ cat << EOU
CSGCopyTest.sh
================

::

   ~/o/CSG/tests/CSGCopyTest.sh


EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

name=CSGCopyTest

bin=$name
script=$name.py

source $HOME/.opticks/GEOM/GEOM.sh

defarg="info_run_ana"
arg=${1:-$defarg}
vars="BASH_SOURCE GEOM name bin script arg BASE AFOLD BFOLD"

export BASE=/tmp/$USER/opticks/$bin
export AFOLD=$BASE/src
export BFOLD=$BASE/dst

logging(){
   export CSGCopy=INFO
   export CSGFoundry=INFO
}
logging

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
    ./CSGTestRunner.sh $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : pdb error && exit 3
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 4
fi





exit 0

