#!/bin/bash -l 

defarg="info_build_run"
arg=${1:-$defarg}

name=salloc_test 

source $HOME/.opticks/GEOM/GEOM.sh 
export FOLD=/tmp/$name
mkdir -p $FOLD

bin=$FOLD/$name

export BASE=$FOLD
export BASE=/tmp/blyth/opticks/GEOM/$GEOM/ntds3

vars="BASH_SOURCE name FOLD BASE OPTICKS_PREFIX GEOM"


if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc \
          -g \
          -std=c++11 -lstdc++ \
          -I.. \
          -I$OPTICKS_PREFIX/externals/glm/glm \
          -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 

    case $(uname) in 
       Darwin) lldb__ $bin ;;
       Linux) gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 


if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 2 
fi 

exit 0 

