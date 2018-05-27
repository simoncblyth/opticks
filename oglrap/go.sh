#!/bin/bash -l

opticks-

sdir=$(pwd)
name=$(basename $sdir)

bdir=$(opticks-prefix)/build/$name

if [ "$1" == "clean" ]; then
   echo $0 $1 : remove bdir $bdir 
   rm -rf $bdir 
fi
mkdir -p $bdir && cd $bdir && pwd 

cmake $sdir \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
    -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
    -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules 

make
[ "$(uname)" == "Darwin" ] && echo kludge 2s sleep && sleep 2 
make install   

opticks-t $bdir



