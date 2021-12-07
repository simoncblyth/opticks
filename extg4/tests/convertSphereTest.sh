#!/bin/bash -l 

opticks-
g4-
clhep-

name=convertSphereTest 
bin=/tmp/$USER/$name
mkdir -p $(dirname $bin)

gcc $name.cc \
    -std=c++11 \
                            \
    -I$(opticks-prefix)/include/sysrap  \
    -I$(opticks-prefix)/include/boostrap  \
    -I$(opticks-prefix)/include/npy  \
    -I$(g4-prefix)/include/Geant4  \
    -I$(clhep-prefix)/include  \
    -I$(opticks-prefix)/externals/glm/glm \
    -I$(opticks-prefix)/externals/plog/include \
                            \
    -L$(g4-prefix)/lib \
    -L$(clhep-prefix)/lib \
    -L$(opticks-prefix)/lib \
                            \
    -lstdc++ \
    -lSysRap \
    -lBoostRap \
    -lNPY \
    -lG4global \
    -lG4materials \
    -lG4geometry \
    -lCLHEP-$(clhep-ver) \
                         \
    -o $bin

[ $? -ne 0 ] && echo compile error && exit 1 


$bin $* 
[ $? -ne 0 ] && echo run error && exit 2 

exit 0 

