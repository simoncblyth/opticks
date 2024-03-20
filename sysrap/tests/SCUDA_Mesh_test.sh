#!/bin/bash -l 
usage(){ cat << EOU
SCUDA_Mesh_test.sh
===================

::

    ~/o/sysrap/tests/SCUDA_Mesh_test.sh 
    ~/o/sysrap/tests/SCUDA_Mesh_test.cc 


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
name=SCUDA_Mesh_test
bin=/tmp/$name

opticks-
glm-

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}
for l in lib lib64 ; do [ -d "$CUDA_PREFIX/$l" ] && cuda_l=$l ; done 

optix_prefix=${OPTICKS_OPTIX_PREFIX}
OPTIX_PREFIX=${OPTIX_PREFIX:-$optix_prefix}

sysrap_dir=..
SYSRAP_DIR=${SYSRAP_DIR:-$sysrap_dir}

scene_fold=/tmp/SScene_test
export SCENE_FOLD=${SCENE_FOLD:-$scene_fold}

vars="BASH_SOURCE CUDA_PREFIX OPTIX_PREFIX cuda_l SCENE_FOLD"
for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done

gcc $name.cc \
    -std=c++11 -lstdc++ -lm -ldl  -g \
    -I${SYSRAP_DIR} \
    -I$CUDA_PREFIX/include \
    -I$OPTIX_PREFIX/include \
    -I$(glm-prefix) \
    -L$CUDA_PREFIX/$cuda_l -lcudart \
    -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 

$bin 
[ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2

exit 0 

