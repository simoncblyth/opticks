#!/bin/bash -l 
usage(){ cat << EOU
CSGIntersectComparisonTest.sh 
===============================

::

    NUM=1000000 ./CSGIntersectComparisonTest.sh 

EOU
}

cd $(dirname $BASH_SOURCE)

bin=CSGIntersectComparisonTest
defarg="run_ana"
arg=${1:-$defarg}

export EPSILON=1e-6

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    case $(uname) in 
       Darwin) lldb__ $bin ;;
       Linux) gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi 

if [ "${arg/ana}" != "$arg" ]; then 

    export FOLD=/tmp/$USER/opticks/$bin
    ${IPYTHON:-ipython} --pdb -i $bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3 
fi 

exit 0 

