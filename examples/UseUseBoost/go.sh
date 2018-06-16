#!/bin/bash -l

opticks-
opticks-id


sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/$name/build 

rm   -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

cmake $sdir \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
    -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
    -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules 

make
make install   

exe=$(opticks-prefix)/lib/$name


if [ -f "$exe" ]; then
    ls -l $exe
    echo running installed exe $exe
    $exe
else 
    echo failed to install exe to $exe 
fi 



