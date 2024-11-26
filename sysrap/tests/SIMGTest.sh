#!/bin/bash -l 

usage(){ cat << EOU
SIMGTest.sh
============

::

    IMGPATH=$HOME/rocket.jpg ~/o/sysrap/tests/SIMGTest.sh 
    IMGPATH=$HOME/rocket.jpg ~/o/sysrap/tests/SIMGTest.sh ana

    P[blyth@localhost ~]$ l /home/blyth/rock*
      8 -rw-rw-r--. 1 blyth blyth   6597 Nov 26 10:28 /home/blyth/rocket_5.jpg
     12 -rw-rw-r--. 1 blyth blyth   8281 Nov 26 10:28 /home/blyth/rocket_10.jpg
     20 -rw-rw-r--. 1 blyth blyth  19210 Nov 26 10:28 /home/blyth/rocket_50.jpg
    228 -rw-rw-r--. 1 blyth blyth 231062 Nov 26 10:28 /home/blyth/rocket_100.jpg
    384 -rw-rw-r--. 1 blyth blyth 391226 Nov 26 10:28 /home/blyth/rocket.png
    112 -rw-rw-r--. 1 blyth blyth 112525 Nov 26 10:27 /home/blyth/rocket.jpg
    P[blyth@localhost ~]$ 


Will not overwrite the loadpath::

    P[blyth@localhost ~]$ IMGPATH=$HOME/rocket.png ~/o/sysrap/tests/SIMGTest.sh run 
    SIMG width 640 height 427 channels 3 loadpath /home/blyth/rocket.png loadext .png
    SIMG::writePNG ERROR cannot overwrite loadpath /home/blyth/rocket.png
    P[blyth@localhost ~]$ 



EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SIMGTest 

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

script=${name}.py

#stb- 

defarg="info_build_run"
arg=${1:-$defarg}

vars="name FOLD bin script"
 

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

if [ "${arg/ana}" != "$arg" ]; then

    [ -z "$IMGPATH" ] && echo $BASH_SOURCE : MISSING IMGPATH OF PNG OR JPG && exit 3

    export NPYPATH=${IMGPATH/.jpg/.npy}
    ${IPYTHON:-ipython} -i --pdb  $script 
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi 




exit 0 


