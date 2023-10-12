#!/bin/bash -l 
usage(){ cat << EOU
build.sh
===========

::

  EYE=0,-4,0,1 ./build.sh run

Using /tmp/sphoton_test/record.npy is convenient for debugging 
as the record array can be defined to have photons moving in 
very simple ways.  


Suspect the Linux CMake build may be finding a GLEW 
lib other than the one installed by opticks-full.
That could cause issues.  
For example there is one as an external of ROOT::

   #glew_prefix=/data/blyth/junotop/ExternalLibs/ROOT/6.24.06
   #GLEW_PREFIX=${GLEW_PREFIX:-$glew_prefix}
   #-Wl,-rpath,$GLEW_PREFIX/lib:$OPTICKS_PREFIX/externals/lib64 \


EOU
}

cd $(dirname $BASH_SOURCE)
name=UseGeometryShader
bdir=/tmp/$name
bin=$bdir/$name
mkdir -p $bdir

sdir=$(pwd)
export SHADER_FOLD=$sdir/rec_flying_point
export ARRAY_FOLD=/tmp/sphoton_test

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}


vars="name bdir sdir SHADER_FOLD ARRAY_FOLD bin"

defarg="info_build_run"
arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then 
    if [ "$(uname)" == "Darwin" ]; then

        echo $BASH_SOURCE : Darwin build
        gcc \
        $name.cc \
        -fvisibility=hidden \
        -fvisibility-inlines-hidden \
        -fdiagnostics-show-option \
        -Wall \
        -Wno-unused-function \
        -Wno-unused-private-field \
        -Wno-shadow \
        -Wsign-compare \
        -g -O0 -std=c++11 \
        -I../../sysrap \
        -I$OPTICKS_PREFIX/externals/glm/glm \
        -I$OPTICKS_PREFIX/externals/include \
        -I$CUDA_PREFIX/include \
        -lstdc++ \
        -L$OPTICKS_PREFIX/externals/lib -lGLEW -lglfw \
        -framework Cocoa \
        -framework OpenGL \
        -framework IOKit \
        -framework CoreFoundation \
        -framework CoreVideo \
        -o $bin
        [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1
        echo $BASH_SOURCE : Darwin build DONE

    elif [ "$(uname)" == "Linux" ]; then

        echo $BASH_SOURCE : Linux building $bin

        gcc \
        $name.cc \
        -fvisibility=hidden \
        -fvisibility-inlines-hidden \
        -fdiagnostics-show-option \
        -Wall \
        -Wno-unused-function \
        -Wno-unused-private-field \
        -Wno-shadow \
        -Wsign-compare \
        -g -O0 -std=c++11 \
        -I../../sysrap \
        -I$OPTICKS_PREFIX/externals/glm/glm \
        -I$OPTICKS_PREFIX/externals/include \
        -I$CUDA_PREFIX/include \
        -lstdc++ \
        -lm \
        -L$OPTICKS_PREFIX/externals/lib -lGLEW \
        -L$OPTICKS_PREFIX/externals/lib64 -lglfw \
        -lGL  \
        -o $bin
    fi
    [ $? -ne 0 ] && echo $BASH_SOURCE : link error && exit 2
fi

if [ "${arg/run}" != "$arg" ]; then
   EYE=0,-3,0,1 $bin
   # both macOS and Linux the ninth ring just gets to top and bottom of render 
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

