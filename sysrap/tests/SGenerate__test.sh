#!/bin/bash
usage(){ cat << EOU
SGenerate__test.sh
====================

CPU test of CUDA code to generate torch photons using srngcpu.h::

   lo # build environment with conda

   ~/o/sysrap/tests/SGenerate__test.sh info_build_run_ls

   MODE=3 ~/o/sysrap/tests/SGenerate__test.sh pdb


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

defarg=info_build_run_ls_ana
arg=${1:-$defarg}

unset SGenerate__GeneratePhotons_RNG_PRECOOKED
#export SGenerate__GeneratePhotons_RNG_PRECOOKED=1

vv="BASH_SOURCE U4TDIR CUDA_PREFIX OPTICKS_PREFIX FOLD bin script SGenerate__test_GS_NAME SGenerate__test_PH_NAME"

#src=sphere
src=disc

if [ "$src" == "sphere" ]; then
    export storch_FillGenstep_type=sphere
    export storch_FillGenstep_radius=100    # +ve for outwards
    export storch_FillGenstep_pos=0,0,0
    export storch_FillGenstep_distance=1.00 # frac_twopi control of polarization phase(tangent direction)

elif [ "$src" == "disc" ]; then

    export storch_FillGenstep_type=disc
    export storch_FillGenstep_radius=399       # 1mm less than sChimneyLS outer radius
    export storch_FillGenstep_zenith=0,1       # radial range scale
    export storch_FillGenstep_azimuth=0,1      # phi segment twopi fraction
    export storch_FillGenstep_mom=0,0,1
    export storch_FillGenstep_pos=0,0,$(( -1963 - 500 ))   # -1963 is base of sChimneyLS

fi

pfx=SGenerate
export SEvent__MakeGenstep_num_ph=M1
#export SEvent__MakeGenstep_num_ph=K1
export SGenerate__test_GS_NAME=${pfx}_gs_${src}_${SEvent__MakeGenstep_num_ph}.npy
export SGenerate__test_PH_NAME=${pfx}_ph_${src}_${SEvent__MakeGenstep_num_ph}.npy


env | grep storch

if [ "$storch_FillGenstep_type" == "" ]; then
    echo $BASH_SOURCE : FATAL : for CHECK $CHECK LAYOUT $LAYOUT GEOM $GEOM
    exit 1
fi

if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%30s : %s \n" "$v" "${!v}" ; done
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

if [ "${arg/ls}" != "$arg" ]; then
    echo ls -alst $FOLD
    ls -alst $FOLD
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

