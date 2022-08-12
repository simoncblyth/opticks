#!/bin/bash -l 

name=sfactor_test 
export FOLD=/tmp/$name
mkdir -p $FOLD

defarg="build_run_ana"
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -g -std=c++11 -lstdc++ -I.. -o /tmp/$name/$name 
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1
fi 

if [ "${arg/run}" != "$arg" ]; then 
   /tmp/$name/$name
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
   lldb__ /tmp/$name/$name
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
   ${IPYTHON:-ipython} --pdb -i $name.py  
   [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4 
fi 

exit 0 



