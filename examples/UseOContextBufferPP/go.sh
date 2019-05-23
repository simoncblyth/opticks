#!/bin/bash -l

opticks-

sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 

echo bdir $bdir name $name

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

  
cmake $sdir -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
            -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
            -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
            -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) 

make
[ $? -ne 0 ] && echo make ERROR && exit 1 

make install   

$name --help

cu_name=bufferTest.cu
ptx=$(opticks-prefix)/installcache/PTX/${name}_generated_${cu_name}.ptx


ls -l $ptx
ptx.py $ptx | c++filt






