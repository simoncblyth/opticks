#!/bin/bash -l
usage(){ cat << EOU
examples/UseOpticksGLFWSnap/go.sh
====================================

Pops up an OpenGL window with a colorful rotating single triangle
On pressing SPACE a ppm snapshot of the window is saved to file.::

   ~/o/examples/UseOpticksGLFWSnap/go.sh

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

echo executing $name
$name


