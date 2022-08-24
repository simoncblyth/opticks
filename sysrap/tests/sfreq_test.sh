#!/bin/bash -l 

defarg="build_run_ana"
arg=${1:-$defarg}


name=sfreq_test 

export FOLD=/tmp/$name/out 
mkdir -p $FOLD


if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -g -std=c++11 -lstdc++ -I.. -o /tmp/$name/$name 
    [ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    /tmp/$name/$name
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    case $(uname) in
       Darwin) lldb__ /tmp/$name/$name ;;
       Linux)  gdb__ /tmp/$name/$name ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3 
fi 


if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i sfreq_test.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 4
fi 



exit 0 


