#!/bin/bash -l 

usage(){ cat << EOU
SIMGTest.sh
============

::

   IMGPATH=/tmp/flower.jpg ~/o/sysrap/tests/SIMGTest.sh 


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SIMGTest 

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

#stb- 

defarg="info_build_run"
arg=${1:-$defarg}

vars="name FOLD bin"
 

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc \
        -lstdc++ -lm -std=c++11 \
        -I$OPTICKS_PREFIX/include/SysRap \
        -o $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then

    [ -z "$IMGPATH" ] && echo $BASH_SOURCE : MISSING IMGPATH OF PNG OR JPG && exit 3
    $bin $IMGPATH
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi 

exit 0 


