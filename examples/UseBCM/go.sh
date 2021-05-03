#!/bin/bash -l

if [ -z "$OPTICKS_PREFIX" -o ! -d "$OPTICKS_PREFIX" ]; then
    echo $BASH_SOURCE : required envvar OPTICKS_PREFIX  $OPTICKS_PREFIX is not defined or not pointing to an existing directory  
    exit 1
fi 
echo OPTICKS_PREFIX : $OPTICKS_PREFIX


sdir=$(pwd)
name=$(basename $sdir) 
PREFIX=/tmp/$USER/opticks/$name
bdir=$PREFIX/build 

rm   -rf $bdir
mkdir -p $bdir 
cd $bdir 
pwd 

cmake $sdir \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_PREFIX_PATH=${OPTICKS_PREFIX}/externals \
    -DCMAKE_INSTALL_PREFIX=$PREFIX 


#    -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules

#make
#make install   



exit 0

