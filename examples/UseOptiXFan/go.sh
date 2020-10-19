#!/bin/bash -l

sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 
idir=/tmp/$USER/opticks/$name/local 

rm -rf $bdir && mkdir -p $bdir 
[ ! -d $bdir ] && exit 1

cd $bdir && pwd 

cuda_prefix=${CUDA_PREFIX:-/usr/local/cuda}
optix_prefix=${OPTIX_PREFIX:-/usr/local/optix}
install_prefix=$idir

cat << EOI

   cuda_prefix    : ${cuda_prefix}
   optix_prefix   : ${optix_prefix}
   install_prefix : ${install_prefix}

EOI



if [ ! -f CMakeCache.txt ]; then 
 
    cmake $sdir \
          -DCMAKE_BUILD_TYPE=Debug \
          -DCMAKE_MODULE_PATH=${optix_prefix}/SDK/CMake \
          -DOptiX_INSTALL_DIR=${optix_prefix} \
          -DCMAKE_INSTALL_PREFIX=${install_prefix}
            
fi 

make
make install   

DYLD_LIBRARY_PATH=${optix_prefix}/lib64:${cuda_prefix}/lib PREFIX=${install_prefix} ${install_prefix}/bin/UseOptiXFan

