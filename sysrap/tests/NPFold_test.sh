#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
name=NPFold_test 
mkdir -p /tmp/$name 

defarg="build_run"
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -std=c++11 -lstdc++ -g -I.. -o /tmp/$name/$name 
    [ $? -ne 0 ] && echo $msg compile error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    /tmp/$name/$name
    [ $? -ne 0 ] && echo $msg run error && exit 2
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    case $(uname) in
      Darwin) lldb__ /tmp/$name/$name ;;
      Linux)  gdb /tmp/$name/$name ;;
    esac
    [ $? -ne 0 ] && echo $msg dbg error && exit 3
fi 

exit 0 

