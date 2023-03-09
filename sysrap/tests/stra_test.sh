#!/bin/bash -l 

name=stra_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I$OPTICKS_PREFIX/externals/glm/glm -o /tmp/$name && /tmp/$name



