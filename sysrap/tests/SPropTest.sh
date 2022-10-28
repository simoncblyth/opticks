#!/bin/bash -l 

name=SPropTest 

defarg="build_run"
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc -std=c++11 -I.. -lstdc++ -I. -o /tmp/$name 
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
    /tmp/$name
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

exit 0 

