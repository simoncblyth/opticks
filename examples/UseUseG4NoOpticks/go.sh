#!/bin/bash 

notes(){ cat << EON


EON
}

sdir=$(pwd)
name=$(basename $sdir)
idir=/tmp/$USER/opticks/$name
bdir=/tmp/$USER/opticks/$name.build 

name0=UseG4NoOpticks
sdir0=$(dirname $sdir)/$name0
idir0=$(dirname $idir)/$name0

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 
rm -rf $idir

env
echo CMAKE_PREFIX_PATH
echo $CMAKE_PREFIX_PATH | tr ":" "\n"

cmake $sdir \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=$idir \
    -DCMAKE_MODULE_PATH=$sdir0


make
make install   




