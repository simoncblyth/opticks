#!/bin/bash -l 

name=fts_test
defarg="build_run"
arg=${1:-$defarg}

tdir=/tmp/NPFold_test
mkdir -p $tdir
mkdir -p $tdir/red
mkdir -p $tdir/green
mkdir -p $tdir/blue

mkdir -p $tdir/red/cyan
mkdir -p $tdir/red/magenta
mkdir -p $tdir/red/yellow

touch $tdir/red/cyan/0
touch $tdir/red/magenta/1
touch $tdir/red/yellow/2


if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name 
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then
    /tmp/$name $tdir
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi

exit 0

