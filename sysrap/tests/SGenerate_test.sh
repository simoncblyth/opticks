#!/bin/bash -l 
usage(){ cat << EOU
SGeneratr_test.sh
================

CPU test of CUDA code to generate torch photons using s_mock_curand.h::

   ./SGenerate_test.sh build
   ./SGenerate_test.sh run
   ./SGenerate_test.sh ana
   ./SGenerate_test.sh build_run_ana   # default 

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
U4TDIR=$(cd $SDIR/../../u4/tests && pwd)

msg="=== $BASH_SOURCE :"
name=SGenerate_test 


cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}


export FOLD=/tmp/$USER/opticks/$name
mkdir -p $FOLD

bin=$FOLD/$name
script=$SDIR/$name.py

defarg=build_run_ana
arg=${1:-$defarg}

vars="BASH_SOURCE SDIR U4TDIR CUDA_PREFIX OPTICKS_PREFIX FOLD bin script"


#check=rain_point_xpositive_100
#check=rain_line
#check=tub3_side_line
#check=circle_inwards_100
#check=circle_outwards_1
check=rain_line_205

export LAYOUT=one_pmt
export CHECK=${CHECK:-$check} 
source $U4TDIR/storch_FillGenstep.sh
echo $BASH_SOURCE : CHECK $CHECK 
env | grep storch

if [ "$storch_FillGenstep_type" == "" ]; then 
    echo $BASH_SOURCE : FATAL : for CHECK $CHECK LAYOUT $LAYOUT GEOM $GEOM 
    exit 1 
fi 



if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then 

    #opt=-DMOCK_CURAND
    opt=-DDUMMY

    gcc $name.cc -std=c++11 -lstdc++ \
           $opt \
           -I.. \
           -I$CUDA_PREFIX/include \
           -I$OPTICKS_PREFIX/externals/glm/glm \
           -I$OPTICKS_PREFIX/externals/plog/include \
           -L$OPTICKS_PREFIX/lib \
           -lSysRap \
           -o $bin

    [ $? -ne 0 ] && echo $msg build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $msg run error && exit 2 
fi

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $msg ana error && exit 3 
fi

exit 0 


