#!/bin/bash -l 

name=s_pool_test 
bin=/tmp/$name

export s_pool_level=1

defarg="build_run"
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc -g -std=c++11 -lstdc++ -I.. -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

if [ "${arg/dbg}" != "$arg" ]; then 
    case $(uname) in
    Darwin) lldb__ $bin ;;
    Linux)  gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi

exit 0 


