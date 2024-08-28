#!/bin/bash 
usage(){ cat << EOU
SSceneLoad_test.sh
===================

Loads persisted SScene and dumps the desc::

   ~/o/sysrap/tests/SSceneLoad_test.sh 

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SScene_test
export FOLD=/tmp/$name
mkdir -p $FOLD 
bin=$FOLD/$name


name=SSceneLoad_test

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

exit 0
