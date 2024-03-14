#!/bin/bash -l
usage(){ cat << EOU
examples/UseShaderSGLFW/go.sh
===============================

Pops up an OpenGL window with a colorful single triangle::

    ~/o/examples/UseShaderSGLFW/go.sh

EOU
}


opticks-
oe-
om-

path=$(realpath $BASH_SOURCE)
sdir=$(dirname $path)
name=$(basename $sdir)

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}
export CUDA_PREFIX

bdir=/tmp/$USER/opticks/$name/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 


defarg=info_build_run
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then
    om-cmake $sdir 
    make
    [ $? -ne 0 ] && echo $BASH_SOURCE : make error && exit 1 

    make install   
    [ $? -ne 0 ] && echo $BASH_SOURCE : install error && exit 2 
fi

if [ "${arg/run}" != "$arg" ]; then
    echo executing $name
    $name
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 3
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    echo executing $name
    dbg__ $name
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 4
fi 

exit 0 

