#!/bin/bash -l
usage(){ cat << EOU
examples/UseOpticksGLFW/go.sh 
===============================

Minimal usage of OpenGL via GLFW : press ESCAPE to exit.

* ancient non-shader OpenGL used. 
* press keys with modifiers to see GLFW key callbacks in action

Based on /Users/blyth/env/graphics/glfw/glfwminimal/glfwminimal.cc

* http://www.glfw.org/docs/latest/quick.html

Usage::

   ~/o/examples/UseOpticksGLFW/go.sh 


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

