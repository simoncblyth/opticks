#!/bin/bash -l
usage(){ cat << EOU
examples/UseOpticksGLFWSPPM/go.sh
==================================

Ancient non-shader OpenGL checking use of SPPM to save the screen buffer when press SPACE::

  ~/o/examples/UseOpticksGLFWSPPM/go.sh

EOU
}

opticks-
oe-
om-

path=$(realpath $BASH_SOURCE)
sdir=$(dirname $path)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

om-cmake $sdir 
make
make install   

echo $0 : executing $name
$name 


