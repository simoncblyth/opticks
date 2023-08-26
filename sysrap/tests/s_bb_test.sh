#!/bin/bash -l 
usage(){ cat << EOU
s_bb_test.sh
=============

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd) 
name=s_bb_test
export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

defarg="info_build_run"
arg=${1:-$defarg}

vars="BASH_SOURCE arg name SDIR FOLD bin"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc \
         $SDIR/$name.cc \
         $SDIR/../s_bb.cc \
         -g -std=c++11 -lstdc++ \
         -I$SDIR/.. \
         -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then
    $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 
fi 
 
exit 0 

