#!/bin/bash
usage(){ cat << EOU
examples/UseShaderSGLFW/go.sh
===============================

Pops up an OpenGL window with a colorful single triangle::

    ~/o/examples/UseShaderSGLFW/go.sh

EOU
}


cd $(dirname $(realpath $BASH_SOURCE))
sdir=$PWD
name=$(basename $sdir)

if [ -z "$OPTICKS_HOME" ]; then
   export OPTICKS_HOME=$(realpath ../..)
   echo $0 - setting OPTICKS_HOME [$OPTICKS_HOME]
else
   echo $0 - using OPTICKS_HOME [$OPTICKS_HOME]
fi 

if [ -z "$CMAKE_PREFIX_PATH" ]; then 
   echo $0 - ERROR - MISSING CMAKE_PREFIX_PATH - NEEDED TO FIND OPTICKS EXTERNALS THIS IS BASED ON  && exit 1 
fi 


cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}
export CUDA_PREFIX

bdir=/tmp/$USER/opticks/$name/build 
idir=/tmp/$USER/opticks/$name/install
mkdir -p $bdir && cd $bdir

PREFIX=$idir
bin=$PREFIX/lib/$name

defarg=info_clean_build_run_check
arg=${1:-$defarg}


if [ "${arg/info}" != "$arg" ]; then
    vv="0 BASH_SOURCE PWD bdir idir defarg arg PREFIX OPTICKS_HOME OPTICKS_PREFIX CUDA_PREFIX bin"
    for v in $vv ; do printf "%20s : %s\n" "$v" "${!v}" ; done
fi 


if [ "${arg/clean}" != "$arg" ]; then
    rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 
    rm -rf $idir && mkdir -p $idir && mkdir -p $idir/lib
fi

if [ "${arg/build}" != "$arg" ]; then
    cmake $sdir \
         -DCMAKE_BUILD_TYPE=Debug \
         -DOPTICKS_PREFIX=$OPTICKS_PREFIX \
         -DCMAKE_INSTALL_PREFIX=$PREFIX \
         -DCMAKE_MODULE_PATH=$OPTICKS_HOME/cmake/Modules

    make
    [ $? -ne 0 ] && echo $BASH_SOURCE : make error && exit 1 

    make install   
    [ $? -ne 0 ] && echo $BASH_SOURCE : install error && exit 2 
fi

if [ "${arg/run}" != "$arg" ]; then
    echo executing $bin
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 3
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    echo executing $bin
    source ../dbg__.sh 
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 4
fi 

if [ "${arg/check}" != "$arg" ]; then 
    echo ls -alst $bin
    ls -alst $bin
    date
    ldd $bin
fi

exit 0 

