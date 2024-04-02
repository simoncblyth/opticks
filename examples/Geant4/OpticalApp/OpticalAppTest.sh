#!/bin/bash -l 
usage(){ cat << EOU

~/o/examples/Geant4/OpticalApp/OpticalAppTest.sh 


EOU
}

cd $(dirname $(realpath $BASH_SOURCE)0)

name=OpticalAppTest
bin=/tmp/$name

vars="BASH_SOURCE name bin"

# -Wno-deprecated-copy \

defarg=info_build_run
arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc \
            -I. \
            -g \
            $(geant4-config --cflags) \
            $(geant4-config --libs) \
             -lstdc++ \
            -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

if [ "${arg/dbg}" != "$arg" ]; then
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 3
fi


exit 0 

