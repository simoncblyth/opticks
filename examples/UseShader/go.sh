#!/bin/bash -l
usage(){ cat << EOU
examples/UseShader/go.sh
=========================

Pops up an OpenGL window with a colorful single triangle::

    ~/o/examples/UseShader/go.sh

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

