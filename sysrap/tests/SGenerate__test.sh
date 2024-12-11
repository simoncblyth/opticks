#!/bin/bash
usage(){ cat << EOU
SGeneratr__test.sh
====================

CPU test of CUDA code to generate torch photons using srngcpu.h::

   ~/o/sysrap/tests/SGenerate__test.sh


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SGenerate__test 

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

tmp=/tmp/$USER/opticks
export TMP=${TMP:-$tmp}
export FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name
script=$name.py

defarg=info_build_run_ana
arg=${1:-$defarg}

export SGenerate__GeneratePhotons_RNG_PRECOOKED=1

vars="BASH_SOURCE U4TDIR CUDA_PREFIX OPTICKS_PREFIX FOLD bin script"

src=sphere
if [ "$src" == "sphere" ]; then
    export storch_FillGenstep_type=sphere
    export storch_FillGenstep_radius=100    # +ve for outwards    
    export storch_FillGenstep_pos=0,0,0
    export storch_FillGenstep_distance=1.00 # frac_twopi control of polarization phase(tangent direction)
fi  

env | grep storch

if [ "$storch_FillGenstep_type" == "" ]; then 
    echo $BASH_SOURCE : FATAL : for CHECK $CHECK LAYOUT $LAYOUT GEOM $GEOM 
    exit 1 
fi 

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then 

    opt=-DNOT_MOCK_CURAND

    gcc $name.cc -std=c++11 -lstdc++ -lm -g \
           $opt \
           -I.. \
           -I$CUDA_PREFIX/include \
           -I$OPTICKS_PREFIX/externals/glm/glm \
           -I$OPTICKS_PREFIX/externals/plog/include \
           -L$OPTICKS_PREFIX/lib64 \
           -lSysRap \
           -o $bin

    [ $? -ne 0 ] && echo $msg build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $msg run error && exit 2 
fi

if [ "${arg/pdb}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $msg pdb error && exit 3 
fi

if [ "${arg/ana}" != "$arg" ]; then 
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $msg ana error && exit 4
fi

exit 0 

