#!/bin/bash -l

opticks-

sdir=$(pwd)
name=$(basename $sdir)

#bdir=/tmp/$USER/opticks/$name/build 
bdir=$(opticks-prefix)/build/$name

rm -rf $bdir 
mkdir -p $bdir && cd $bdir && pwd 

cmake $sdir \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
    -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
    -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
     \
    -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) \
    -DCOMPUTE_CAPABILITY=$(opticks-compute-capability)

## OKConf finds OptiX to get the version integer, so OptiX_INSTALL_DIR is required

make

if [ "$(uname)" == "Darwin" ]; then
   echo "kludge sleeping for 2s"
   sleep 2
fi 

make install   


$(opticks-prefix)/lib/OKConfTest


bcm_deploy_conf=$(opticks-prefix)/lib/cmake/okconf/okconf-config.cmake
ls -l $bcm_deploy_conf
cat $bcm_deploy_conf



