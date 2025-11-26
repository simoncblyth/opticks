#!/bin/bash 

usage(){ cat << EOU

~/o/sysrap/tests/SClientSimulator_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SClientSimulator_test

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
FOLD=$TMP/$name
mkdir -p $FOLD
bin=$FOLD/$name
vv="BASH_SOURCE PWD name tmp TMP FOLD bin"

defarg="info_gcc_run"
arg=${1:-$defarg}
vv="$vv defarg arg"

LIBCURL_VERSION=$(curl-config --version)
vv="$vv LIBCURL_VERSION"

if [ "${arg/info}" != "${arg}" ]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi 

if [ "${arg/gcc}" != "${arg}" ]; then
    gcc $name.cc \
        -o $bin \
        -std=c++17 -lstdc++ \
        -I.. \
        $(curl-config --cflags) \
        $(curl-config --libs)

    [ $? -ne 0 ] && echo $BASH_SOURCE gcc error && exit 1
fi 

if [ "${arg/run}" != "${arg}" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi 

exit 0

