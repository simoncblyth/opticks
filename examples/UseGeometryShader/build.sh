#!/bin/bash -l 
usage(){ cat << EOU
build.sh
===========

::

   EYE=0,-4,0,1 ./build.sh run

Using /tmp/sphoton_test/record.npy is convenient for debugging 
as the record array can be defined to have photons moving in 
very simple ways.  

EOU
}

name=UseGeometryShader
bdir=/tmp/$name
bin=$bdir/$name
mkdir -p $bdir

sdir=$(pwd)
export SHADER_FOLD=$sdir/rec_flying_point
#export ARRAY_FOLD=/tmp/blyth/opticks/GEOM/V1J011/ntds3/ALL1/p001
export ARRAY_FOLD=/tmp/sphoton_test

vars="name bdir sdir SHADER_FOLD"

defarg="info_build_run"
arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then 
   gcc \
     -I../../sysrap -I/usr/local/cuda/include \
     -isystem $OPTICKS_PREFIX/externals/glm/glm \
     -isystem $OPTICKS_PREFIX/externals/include \
     -fvisibility=hidden \
     -fvisibility-inlines-hidden \
     -fdiagnostics-show-option \
     -Wall \
     -Wno-unused-function \
     -Wno-unused-private-field \
     -Wno-shadow \
     -Wsign-compare \
     -g -O0 -std=c++11 -o $bdir/$name.o \
     -c $name.cc

   [ $? -ne 0 ] && echo $BASH_SOURCE : compile error && exit 1 

   gcc \
     -fvisibility=hidden \
     -fvisibility-inlines-hidden \
     -fdiagnostics-show-option \
     -Wall \
     -Wno-unused-function \
     -Wno-unused-private-field \
     -Wno-shadow \
     -Wsign-compare \
      -g -O0 \
      -lstdc++ \
      $bdir/$name.o -o $bin \
      $OPTICKS_PREFIX/externals/lib/libGLEW.dylib \
      $OPTICKS_PREFIX/externals/lib/libglfw.dylib \
      -framework Cocoa \
      -framework OpenGL \
      -framework IOKit \
      -framework CoreFoundation \
      -framework CoreVideo 
     # TODO: Linux equiv 

   [ $? -ne 0 ] && echo $BASH_SOURCE : link error && exit 2
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 3
fi 

if [ "${arg/dbg}" != "$arg" ]; then
   case $(uname) in
      Darwin) lldb__ $bin ;;
      Linux)  gdb__ $bin ;;
   esac
   [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 4
fi 

exit 0 

