#!/bin/bash -l 
usage(){ cat << EOU
build.sh
===========

Usage
------

::

  ~/o/examples/UseGeometryShader/build.sh 
  EYE=0,-4,0,1 ~/o/examples/UseGeometryShader/build.sh run

  RECORD_FOLD=/data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL/p003 ~/o/examples/UseGeometryShader/build.sh 
  RECORD_FOLD=/data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL/p003 ~/o/examples/UseGeometryShader/build.sh  ana

HMM: for debug need a mode that draws all points 


OpenGL graphics externals : GLEW, GLFW
----------------------------------------

As the graphics externals are not currently included in standard build, 
need to manually get and install them with::

    glew-
    glew--

    glfw-
    glfw--

Prepare record array
----------------------

Rather than using a real record array it is 
convenient for debugging to use a fabricated one 
with photons with step points propagating 
in a known simple shape : expanding concentric circles.
Create that record array with::

    TEST=make_record_array ~/o/sysrap/tests/sphoton_test.sh

The extent of step point positions and times are (10 "mm", 10 "ns")

Linux issue ?
--------------

Suspect the Linux CMake build may be finding a GLEW 
lib other than the one installed by opticks-full.
That could cause issues.  
For example there is one as an external of ROOT::

   #glew_prefix=/data/blyth/junotop/ExternalLibs/ROOT/6.24.06
   #GLEW_PREFIX=${GLEW_PREFIX:-$glew_prefix}
   #-Wl,-rpath,$GLEW_PREFIX/lib:$OPTICKS_PREFIX/externals/lib64 \


Commands : info, build, run, dbg, examine
--------------------------------------------

examine 
    uses otool/ldd to dump the libraries linked from the executable


About the render
----------------

With default "EYE=0,-3,0,1" in perspective rendering on both macOS and Linux 
the ninth ring just gets to top and bottom of render. 
BUT: the circle appears slightly elliptical : longer vertically that horizontally 
when would expect equal. 

Is the aspect ratio calc in the view math using correct screen size, accounting for "chrome" ?

TODO: orthographic mode switch  


EOU
}

defarg="info_build_run_examine"
arg=${1:-$defarg}

cd $(dirname $(realpath $BASH_SOURCE))

name=UseGeometryShader
bdir=/tmp/$name
bin=$bdir/$name
script=$name.py 

mkdir -p $bdir

sdir=$(pwd)

#shader=rec_flying_point
shader=rec_flying_point_persist
#shader=pos
SHADER=${SHADER:-$shader}
export SHADER_FOLD=$sdir/$SHADER
#export SHADER_FOLD=$SHADER

record_fold=/tmp/sphoton_test
export RECORD_FOLD=${RECORD_FOLD:-$record_fold}

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

eye=0,-3,0
EYE=${EYE:-$eye}
export EYE

vars="BASH_SOURCE PWD name bdir sdir CUDA_PREFIX OPTICKS_PREFIX SHADER_FOLD RECORD_FOLD bin EYE"

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

        echo $BASH_SOURCE : Linux build $bin BEGIN

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
        [ $? -ne 0 ] && echo $BASH_SOURCE : Linux build error && exit 2
        echo $BASH_SOURCE : Linux build $bin DONE
    fi
fi

if [ "${arg/run}" != "$arg" ]; then

   if [ "$(uname)" == "Darwin" ]; then

      echo $BASH_SOURCE : Darwin running $bin : benefits from RPATH ? 
      $bin
      [ $? -ne 0 ] && echo $BASH_SOURCE : Darwin run error && exit 3

    elif [ "$(uname)" == "Linux" ]; then

      LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPTICKS_PREFIX/externals/lib:$OPTICKS_PREFIX/externals/lib64
      echo $BASH_SOURCE : Linux running $bin : with some manual LD_LIBRARY_PATH config 
      $bin
      [ $? -ne 0 ] && echo $BASH_SOURCE : Linux run error && exit 3

    fi

fi 

if [ "${arg/dbg}" != "$arg" ]; then
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 4
fi 

if [ "${arg/examine}" != "$arg" ]; then 
   echo $BASH_SOURCE : examine libs of the binary $bin
   case $(uname) in
      Darwin) otool -L $bin ;;
      Linux)  ldd $bin ;;
   esac
   [ $? -ne 0 ] && echo $BASH_SOURCE : examine error && exit 5
fi 

if [ "${arg/ana}" != "$arg" ]; then
   ${IPYTHON:-ipython} --pdb -i  $script
   [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 6
fi 




exit 0 

