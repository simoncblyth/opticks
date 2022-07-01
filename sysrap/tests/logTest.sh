#!/bin/bash -l 
usage(){ cat << EOU
logTest.sh : comparing CUDA __logf with logf without -use_fast_math
======================================================================

When using option -use_fast_math logf becomes __logf so no differences are visible.::

    cd ~/opticks/sysrap/tests

    ./logTest.sh build_run_ana   # default 
    ./logTest.sh build
    ./logTest.sh run
    ./logTest.sh ana
    ./logTest.sh grab

    UNAME=Linux ./logTest.sh ana


EOU
}

msg="=== $BASH_SOURCE : "
name=logTest 

defarg="build_run_ana"
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then 

    #opt="-use_fast_math"
    opt="" 
    echo $msg opt $opt
    nvcc $name.cu -std=c++11 $opt -I.. -I/usr/local/cuda/include -o /tmp/$name 
    [ $? -ne 0 ] && echo compilation error && exit 1
fi 

base=/tmp/$USER/opticks/sysrap/logTest
UNAME=${UNAME:-$(uname)}

export FOLD=$base/$UNAME
mkdir -p $FOLD
echo $msg UNAME $UNAME FOLD $FOLD

if [ "${arg/run}" != "$arg" ]; then 
    /tmp/$name
    [ $? -ne 0 ] && echo run  error && exit 2
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo ana error && exit 3
fi 

if [ "${arg/grab}" != "$arg" ]; then 
    rsync -av P:$base/ $base 
    [ $? -ne 0 ] && echo grab error && exit 4
fi 



exit 0 

