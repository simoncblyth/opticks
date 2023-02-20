#!/bin/bash -l 

name=sn_test

export FOLD=/tmp/$name
mkdir -p $FOLD
 
bin=$FOLD/$name
defarg="build_run"
arg=${1:-$defarg}

opt=-DWITH_CHILD
export s_pool_level=2


if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -I.. -I$HOME/np $opt -g -std=c++11 -lstdc++ -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    case $(uname) in 
    Linux)   gdb__ $bin ;;
    Darwin) lldb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi 


exit 0 

