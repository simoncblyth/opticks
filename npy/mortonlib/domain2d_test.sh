#!/bin/bash -l 

arg=${1:-build_run_ana}

name=domain2d_test 
export FOLD=/tmp/$name
mkdir -p $FOLD

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc -g -std=c++11 -lstdc++ -I. -I$HOME/np -o /tmp/$name/$name 
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then
    /tmp/$name/$name
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython } --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3 
fi 

exit 0 





