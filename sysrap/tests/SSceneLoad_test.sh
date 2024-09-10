#!/bin/bash 
usage(){ cat << EOU
SSceneLoad_test.sh
===================

Loads persisted SScene and dumps the desc::

   ~/o/sysrap/tests/SSceneLoad_test.sh 

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SSceneLoad_test
export FOLD=/tmp/$name
mkdir -p $FOLD 
bin=$FOLD/$name

source $HOME/.opticks/GEOM/GEOM.sh 

export SCENE_FOLD=$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim

vars="BASH_SOURCE PWD SCENE_FOLD GEOM CUDA_PREFIX GLM_PREFIX OPTICKS_PREFIX"


defarg=info_build_run
arg=${1:-$defarg}

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

glm_prefix=$OPTICKS_PREFIX/externals/glm/glm
GLM_PREFIX=${GLM_PREFIX:-$glm_prefix}


if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi   

if [ "${arg/build}" != "$arg" ]; then 

    echo $BASH_SOURCE [ building 
    gcc  $name.cc \
         -std=c++11 -lstdc++ -lm \
         -I.. -g \
         -I$GLM_PREFIX \
         -I${CUDA_PREFIX}/include \
         -DWITH_CHILD \
         -o $bin

    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1
    echo $BASH_SOURCE ] building 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi 

gdb__ () 
{ 
    if [ -z "$BP" ]; then
        H="";
        B="";
        T="-ex r";
    else
        H="-ex \"set breakpoint pending on\"";
        B="";
        for bp in $BP;
        do
            B="$B -ex \"break $bp\" ";
        done;
        T="-ex \"info break\" -ex r";
    fi;
    local runline="gdb $H $B $T --args $* ";
    echo $runline;
    date;
    eval $runline;
    date
}

if [ "${arg/dbg}" != "$arg" ]; then 
    gdb__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi 





exit 0
