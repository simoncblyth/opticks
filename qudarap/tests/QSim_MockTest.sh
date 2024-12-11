#!/bin/bash
usage(){ cat << EOU
QSim_MockTest.sh
==================

~/o/qudarap/tests/QSim_MockTest.sh

Note that unlike the standard Opticks CMake build
this script finds Custom4 without consulting 
the CMAKE_PREFIX_PATH so its convenient for 
testing new versions.  

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
source dbg__.sh

name=QSim_MockTest

source $HOME/.opticks/GEOM/GEOM.sh 

defarg="info_build_run_ana"
arg=${1:-$defarg}

export BASE=/tmp/$name
bin=$BASE/$name

#custom4_prefix=${OPTICKS_PREFIX}_externals/custom4/0.1.8
custom4_prefix=$JUNOTOP/ExternalLibs/custom4/0.1.8
CUSTOM4_PREFIX=${CUSTOM4_PREFIX:-$custom4_prefix}


cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

#num=1000
num=100000
#num=1000000
export NUM=${NUM:-$num}

check=smear_normal_sigma_alpha
#check=smear_normal_polish
export CHECK=${CHECK:-$check}

export FOLD=$BASE/$CHECK
mkdir -p $FOLD

case $CHECK in 
   smear_normal*) script=smear_normal.py ;;
               *) script=$name.py        ;;
esac


vars="BASH_SOURCE BASE FOLD CHECK GEOM bin script name CUSTOM4_PREFIX OPTICKS_PREFIX CUDA_PREFIX NUM"


if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/check}" != "$arg" ]; then
    path="${CUSTOM4_PREFIX}/include/Custom4/C4CustomART.h"
    if [ -f "$path" ]; then
       echo $BASH_SOURCE : path $path : EXISTS
    else
       echo $BASH_SOURCE : path $path : DOES NOT EXIST
    fi 
fi 

# gives exponent has no digits error with c++11

if [ "${arg/build}" != "$arg" ]; then 

    [ ! -d "$CUSTOM4_PREFIX" ] && echo $BASH_SOURCE ERROR CUSTOM4_PREFIX $CUSTOM4_PREFIX DOES NOT EXIST && exit 1

    gcc $name.cc \
       ../QPMT.cc \
       ../QOptical.cc \
       ../QBnd.cc \
       ../QTex.cc \
       ../QProp.cc \
       ../QBase.cc \
       -g \
       -std=c++17 -lstdc++ -lm \
       -DDEBUG_PIDX \
       -DMOCK_CURAND \
       -DMOCK_CUDA \
       -DMOCK_CUDA_DEBUG \
       -DMOCK_CURAND_DEBUG \
       -DMOCK_TEXTURE \
       -I.. \
       -I$OPTICKS_PREFIX/include/SysRap \
       -I$CUDA_PREFIX/include \
       -I$OPTICKS_PREFIX/externals/glm/glm \
       -I$OPTICKS_PREFIX/externals/plog/include \
       -DWITH_CUSTOM4 \
       -I$CUSTOM4_PREFIX/include/Custom4 \
       -o $bin 

    [ $? -ne 0 ] && echo $msg build error && exit 1 
fi 


if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $msg run error && exit 2 
fi

if [ "${arg/dbg}" != "$arg" ]; then 
    dbg__ $bin
    [ $? -ne 0 ] && echo $msg dbg error && exit 3 
fi

if [ "${arg/pdb}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $msg pdb error && exit 4
fi

if [ "${arg/ana}" != "$arg" ]; then 
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $msg ana error && exit 4
fi


exit 0 


