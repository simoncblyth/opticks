#!/bin/bash -l 

msg="=== $BASH_SOURCE :"

clhep-
g4-

name=G4RandomDirectionTest
export NPY_PATH=/tmp/$name/marsaglia.npy

arg=${1:-build_run_ana}

if [ "${arg/build}" != "$arg" ]; then 
    mkdir -p /tmp/$name
    gcc $name.cc \
       -std=c++11 \
       -lstdc++ \
       -I$HOME/np \
       -I$(clhep-prefix)/include \
       -I$(g4-prefix)/include/Geant4  \
       -L$(clhep-prefix)/lib \
       -lCLHEP \
        -o /tmp/$name/$name

    [ $? -ne 0 ] && echo $msg compile error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    /tmp/$name/$name
    [ $? -ne 0 ] && echo $msg run error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} -i $name.py 
    [ $? -ne 0 ] && echo $msg ana error && exit 3 
fi 

exit 0 



