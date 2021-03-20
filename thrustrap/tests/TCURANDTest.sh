#!/bin/bash

name=TCURANDTest
tmp=/tmp/$USER/opticks
mkdir -p $tmp

gcc $name.cc -lstdc++ \
    -I.. \
    -I$OPTICKS_PREFIX/externals/plog/include \
    -I$OPTICKS_PREFIX/externals/glm/glm \
    -I$OPTICKS_PREFIX/include/SysRap \
    -I$OPTICKS_PREFIX/include/BoostRap \
    -I$OPTICKS_PREFIX/include/NPY \
    -I$OPTICKS_PREFIX/include/OpticksCore \
    -L$OPTICKS_PREFIX/lib64 \
    -lSysRap \
    -lBoostRap \
    -lNPY \
    -lOpticksCore \
    -lThrustRap \
    -std=c++11 \
    -o $tmp/$name


$tmp/$name
