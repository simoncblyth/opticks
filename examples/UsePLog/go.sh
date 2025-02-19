#!/bin/bash


sdir=$(pwd)
name=$(basename $sdir) 
bdir=/tmp/$USER/opticks/$name/build 

rm   -rf $bdir
mkdir -p $bdir 
cd $bdir 
pwd 

cmake $sdir \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_PREFIX_PATH=$OPTICKS_PREFIX/externals \
    -DCMAKE_INSTALL_PREFIX=$OPTICKS_PREFIX \
    -DCMAKE_MODULE_PATH=$OPTICKS_PREFIX/cmake/Modules \
    -DOPTICKS_PREFIX=$OPTICKS_PREFIX

make
make install   



