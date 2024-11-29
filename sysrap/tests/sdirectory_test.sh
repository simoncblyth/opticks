#!/bin/bash 

usage(){ cat << EOU
sdirectory_test.sh
===================

~/o/sysrap/tests/sdirectory_test.sh


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=sdirectory_test

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

#test=MakeDirs
test=DirList
export TEST=${TEST:-$test}


defarg="info_build_run"
arg=${1:-$defarg}

vars="BASH_SOURCE defarg arg name FOLD bin test TEST"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi 

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc -std=c++11 -lstdc++ -g -I.. -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi




