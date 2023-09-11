#!/bin/bash -l 

name=S4Random_test
cd $(dirname $BASH_SOURCE)



clhep-
g4-

defarg="info_build_run"
arg=${1:-$defarg}

vars="BASH_SOURCE arg"

FOLD=/tmp/$name
mkdir -p $FOLD

bin=$FOLD/$name


if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc \
        -std=c++11 -lstdc++ \
        -I.. \
        -I$(clhep-prefix)/include \
        -I$(g4-prefix)/include/Geant4  \
        -L$(clhep-prefix)/lib \
        -lCLHEP \
        -o $bin 

    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then 
    $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 
fi

exit 0 


