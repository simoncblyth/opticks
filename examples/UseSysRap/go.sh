#!/bin/bash 
usage(){ cat << EOU

~/opticks/examples/UseSysRap/go.sh

EOU
}


cd $(dirname $BASH_SOURCE)

sdir=$(pwd)
bdir=/tmp/$USER/opticks/$(basename $sdir)/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

vars="0 BASH_SOURCE sdir bdir OPTICKS_PREFIX"
for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done

cmake $sdir \
     -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_INSTALL_PREFIX=$OPTICKS_PREFIX \
     -DCMAKE_PREFIX_PATH=$OPTICKS_PREFIX/externals \
     -DCMAKE_MODULE_PATH=$sdir/../../cmake/Modules \
     -DOPTICKS_PREFIX=$OPTICKS_PREFIX

make
make install   

