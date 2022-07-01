#!/bin/bash -l 

msg="=== $BASH_SOURCE : "
name=logTest 

defarg="build_run_ana"
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then 

    #opt="-use_fast_math"
    opt="" 
    echo $msg opt $opt
    nvcc $name.cu -std=c++11 $opt -I.. -I/usr/local/cuda/include -o /tmp/$name 
    [ $? -ne 0 ] && echo compilation error && exit 1
fi 

if [ "${arg/run}" != "$arg" ]; then 
    /tmp/$name
    [ $? -ne 0 ] && echo run  error && exit 2
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo ana error && exit 3
fi 

exit 0 

