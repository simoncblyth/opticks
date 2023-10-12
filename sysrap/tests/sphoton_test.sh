#!/bin/bash -l 
usage(){ cat << EOU
sphoton_test.sh 
==================

EOU
}

REALDIR=$(cd $(dirname $BASH_SOURCE) && pwd )
name=sphoton_test 

defarg="info_build_run_ana"
arg=${1:-$defarg}

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name 

vars="BASH_SOURCE REALDIR FOLD name bin"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -std=c++11 -lstdc++ -lm -lcrypto -lssl \
           -I.. \
           -I/usr/local/cuda/include \
           -I$OPTICKS_PREFIX/externals/glm/glm \
           -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $REALDIR/$name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
fi 


exit 0 


