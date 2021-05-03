#!/bin/bash -l
opticks-


sdir=$(pwd)

name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 

rm -rf $bdir && mkdir -p $bdir 
[ ! -d $bdir ] && exit 1
cd $bdir && pwd 

om-

echo " OPTICKS_OPTIX_PREFIX : ${OPTICKS_OPTIX_PREFIX} "

om-cmake $sdir
[ $? -ne 0 ] && echo config error && exit 1

make
[ $? -ne 0 ] && echo make error && exit 2

make install   
[ $? -ne 0 ] && echo install error && exit 3

which $name && $name
[ $? -ne 0 ] && echo runtime error && exit 4


exit 0 
