#!/bin/bash -l
usage(){ cat << EOU
examples/UseOpticksGLEW/go.sh
===============================

Trivial GLEW CMake and GLEW version macro test::

    ~/o/examples/UseOpticksGLEW/go.sh


EOU
}


opticks-

path=$(realpath $BASH_SOURCE)
sdir=$(dirname $path)
name=$(basename $sdir)

bdir=/tmp/$USER/opticks/$(basename $sdir)/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 


#cmake $sdir -DCMAKE_BUILD_TYPE=Debug \
#            -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
#            -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
#            -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules 

om-
om-cmake $sdir

make
make install   

$name


