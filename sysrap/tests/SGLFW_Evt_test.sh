#!/bin/bash
usage(){ cat << EOU
SGLFW_Evt_test.sh : triangulated raytrace and rasterized visualization
=================================================================================

~/o/sysrap/tests/SGLFW_Evt_test.sh

MOI=sChimneyAcrylic:0:-1 EYE=0,10000,0 T0=100 ~/o/sysrap/tests/SGLFW_Evt_test.sh run

MOI=sChimneyLS:0:-1 T1=10 NT=1000 ~/o/sysrap/tests/SGLFW_Evt_test.sh run

MOI=sChimneyLS:0:-1 T1=30 NT=1000 TIMESCALE=10 ~/o/sysrap/tests/SGLFW_Evt_test.sh run


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
name=SGLFW_Evt_test


source $HOME/.opticks/GEOM/GEOM.sh
[ -z "$GEOM" ] && echo $BASH_SOURCE FATAL GEOM $GEOM IS REQUIRTED && exit 1

source $HOME/.opticks/GEOM/EVT.sh   ## optionally sets AFOLD BFOLD where event info is loaded from
source $HOME/.opticks/GEOM/MOI.sh   ## optionally sets MOI envvar controlling initial viewpoint


export FOLD=/tmp/$USER/opticks/$name
bin=$FOLD/$name
mkdir -p $FOLD


cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}
for l in lib lib64 ; do [ -d "$CUDA_PREFIX/$l" ] && cuda_l=$l ; done


sysrap_dir=..
SYSRAP_DIR=${SYSRAP_DIR:-$sysrap_dir}




logging(){
    type $FUNCNAME
    export SGLM__set_frame_DUMP=1
    export SGLM__setTreeScene_DUMP=1
    export SGLFW_Evt_test_DUMP=1
}
[ -n "$LOG" ] && logging



#wh=1024,768
wh=2560,1440


#eye=-1,-1,0
eye=0,-3,0
up=0,0,1
look=0,0,0

cam=perspective
#cam=orthographic

#fullscreen=0
fullscreen=1

tmin=0.1
#escale=asis
escale=extent


export WH=${WH:-$wh}
export FULLSCREEN=${FULLSCREEN:-$fullscreen}
export EYE=${EYE:-$eye}
export LOOK=${LOOK:-$look}
export UP=${UP:-$up}
export TMIN=${TMIN:-$tmin}
export ESCALE=${ESCALE:-$escale}
export CAM=${CAM:-$cam}


zoom=1
export ZOOM=${ZOOM:-$zoom}


#soptix_handle=-1 # IAS
#soptix_handle=0  # GAS0 global
#soptix_handle=1  # GAS1
#export SOPTIX_HANDLE=${SOPTIX_HANDLE:-$soptix_handle}
## NB SOPTIX is for optix handled triangles : thats not yet implemented for this apph


vizmask=t    # "t"  not everything ie 0xff : no masking
#vizmask=t0  # "t0" not 0-th bit ie 0xfe : ie mask global
export VIZMASK=${VIZMASK:-$vizmask}



defarg="info_build_run"
arg=${1:-$defarg}

[ -n "$BP" ] && defarg="dbg"


PATH=$PATH:$CUDA_PREFIX/bin

vars="BASH_SOURCE defarg arg CUDA_PREFIX cuda_l SCENE_FOLD FOLD bin SGLFW_FRAME SOPTIX_HANDLE"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done
fi

if [ "${arg/ls}" != "$arg" ]; then
   ff="AFOLD BFOLD FOLD"
   for f in $ff ; do printf "%20s : ls -alst %s\n" "$f" "${!f}" ; ls -alst ${!f} ; done
fi


if [ "${arg/build}" != "$arg" ]; then

    echo $BASH_SOURCE build
    [ "$(uname)" == "Darwin" ] && echo $BASH_SOURCE : ERROR : THIS NEEDS OPTIX7+ SO LINUX ONLY && exit 1

    # -M lists paths of all included headers in Makefile dependency format
    # -M \
    gcc $name.cc \
        -fvisibility=hidden \
        -fvisibility-inlines-hidden \
        -fdiagnostics-show-option \
        -Wall \
        -Wno-unused-function \
        -Wno-shadow \
        -Wsign-compare \
        -DWITH_CUDA_GL_INTEROP \
        -DWITH_CHILD \
        -g -O0 -std=c++17 \
        -I$SYSRAP_DIR \
        -I$OPTICKS_PREFIX/externals/glm/glm \
        -I$OPTICKS_PREFIX/externals/include \
        -I$OPTICKS_PREFIX/include/SysRap \
        -I$CUDA_PREFIX/include \
        -L$CUDA_PREFIX/$cuda_l -lcudart \
        -lstdc++ \
        -lm -ldl \
        -L$OPTICKS_PREFIX/lib64 -lSysRap \
        -L$OPTICKS_PREFIX/externals/lib -lGLEW \
        -L$OPTICKS_PREFIX/externals/lib64 -lglfw \
        -lGL  \
        -o $bin



    # -Wno-unused-private-field \  ## clang-ism ?

    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1
    echo $BASH_SOURCE build DONE
fi

if [ "${arg/dbg}" != "$arg" ]; then
    source dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

if [ "${arg/run}" != "$arg" ]; then
    [ -z "$DISPLAY" ] && echo $BASH_SOURCE adhoc setting DISPLAY && export DISPLAY=:0
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 3
fi


exit 0

