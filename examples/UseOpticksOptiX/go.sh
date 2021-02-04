#!/bin/bash -l
opticks-


sdir=$(pwd)

name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 

rm -rf $bdir && mkdir -p $bdir 
[ ! -d $bdir ] && exit 1

cd $bdir && pwd 

om-


## dirty change to the standard CMAKE_PREFIX_PATH setup by om-
## for easy flipping from standard 6 to experimental 7
## which will invalidate the libs built against optix6
#CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH/$OPTICKS_OPTIX_PREFIX/$OPTICKS_OPTIX7_PREFIX}
#echo $CMAKE_PREFIX_PATH | tr ":" "\n"

## ahh : cmake/Modules/FindOpticksOptix.cmake  is finding the optix.h header simply from the envvar OPTICKS_OPTIX_PREFIX
## and totally ignoring CMAKE_PREFIX_PATH
OPTICKS_OPTIX_PREFIX=$OPTICKS_OPTIX7_PREFIX

om-cmake $sdir

make
make install   




