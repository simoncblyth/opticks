#!/bin/bash -l 
usage(){ cat << EOU
SGLM_SmoothNormals_test.sh
==========================

::

   ~/o/sysrap/tests/SGLM_SmoothNormals_test.sh


EOU
}


defarg="build_run_info"
arg=${1:-$defarg}

br="------------------------------------------------------------------------------"
msg="=== $BASH_SOURCE :"
name=SGLM_SmoothNormals_test 
bin=/tmp/$name

cd $(dirname $(realpath $BASH_SOURCE))

mesh_fold=/tmp/U4Mesh_test
MESH_FOLD=${MESH_FOLD:-$mesh_fold}   
export MESH_FOLD

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -g -Wall -std=c++11 -lstdc++ -I.. \
             -I$OPTICKS_PREFIX/externals/glm/glm \
             -I$CUDA_PREFIX/include -o $bin
    [ $? -ne 0 ] && echo $msg build error && exit 1 
    echo $BASH_SOURCE : build OK
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $msg run error && exit 2 
    echo $BASH_SOURCE : run OK
fi 

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
    echo $BASH_SOURCE : info
fi 

exit 0 

