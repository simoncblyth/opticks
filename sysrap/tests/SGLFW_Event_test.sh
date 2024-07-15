#!/bin/bash 
usage(){ cat << EOU
SGLFW_Event_test.sh : triangulated raytrace and rasterized visualization
=================================================================================




EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
name=SGLFW_Event_test


#source $HOME/.opticks/GEOM/GEOM.sh
#[ -z "$GEOM" ] && echo $BASH_SOURCE FATAL GEOM $GEOM IS REQUIRTED && exit 1 


export FOLD=/tmp/$USER/opticks/$name
bin=$FOLD/$name
mkdir -p $FOLD


# when using CMake generated ptx will be smth like:$OPTICKS_PREFIX/ptx/sysrap_generated_SOPTIX.cu.ptx 
# following pattern $OPTICKS_PREFIX/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx" 

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}
for l in lib lib64 ; do [ -d "$CUDA_PREFIX/$l" ] && cuda_l=$l ; done 

#optix_prefix=${OPTICKS_OPTIX_PREFIX}
#OPTIX_PREFIX=${OPTIX_PREFIX:-$optix_prefix}

sysrap_dir=..
SYSRAP_DIR=${SYSRAP_DIR:-$sysrap_dir}

 
scene=1
case ${SCENE:-$scene} in 
0) 
    scene_fold=$HOME/.opticks/GEOM/RaindropRockAirWater/CSGFoundry/SSim
    record_path=/tmp/ihep/opticks/GEOM/RaindropRockAirWater/G4CXTest/ALL0/A000/record.npy 
    ;;
1) scene_fold=$HOME/.opticks/GEOM/J23_1_0_rc3_ok0/CSGFoundry/SSim 
    record_path=/tmp/ihep/opticks/GEOM/J23_1_0_rc3_ok0/jok-tds/ALL0/A000/record.npy
    ;;
esac
export SCENE_FOLD=${SCENE_FOLD:-$scene_fold}
export SRECORD_PATH=${SRECORD_PATH:-$record_path}


shader_fold=../../examples/UseShaderSGLFW_SScene_encapsulated/gl
export SHADER_FOLD=${SHADER_FOLD:-$shader_fold}

export RECORDER_SHADER_FOLD=../../examples/UseGeometryShader/rec_flying_point_persist

dump=0
DUMP=${DUMP:-$dump}
export SGLM__set_frame_DUMP=$DUMP

export SGLFW_Event_test_DUMP=1  


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
#handle=-1 # -1:IAS 0...8 GAS indices 
#export HANDLE=${HANDLE:-$handle}
#:
#frame=-1
#export SGLFW_FRAME=${SGLFW_FRAME:-$frame}

#vizmask=t   # 0xff no masking
#vizmask=t0  # 0xfe mask global
#export VIZMASK=${VIZMASK:-$vizmask}


#export SOPTIX_SBT__initHitgroup_DUMP=1



defarg="info_build_run"
arg=${1:-$defarg}



if [ ! -d "$SCENE_FOLD/scene" ]; then
  echo $BASH_SOURCE : FATAL SCENE_FOLD $SCENE_FOLD does not contain scene
  echo $BASH_SOURCE : with newly created CSGFoundry/SSim there is no need for manual SCENE_FOLD as will be in CSGFoundry/SSim/scene  
  #arg=info  
fi 



PATH=$PATH:$CUDA_PREFIX/bin

vars="BASH_SOURCE defarg arg CUDA_PREFIX cuda_l SCENE_FOLD FOLD bin SGLFW_FRAME"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done
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
        -g -O0 -std=c++11 \
        -I$SYSRAP_DIR \
        -I$OPTICKS_PREFIX/externals/glm/glm \
        -I$OPTICKS_PREFIX/externals/include \
        -I$OPTICKS_PREFIX/include/SysRap \
        -I$CUDA_PREFIX/include \
        -L$CUDA_PREFIX/$cuda_l -lcudart \
        -lstdc++ \
        -lm -ldl \
        -L$OPTICKS_PREFIX/externals/lib -lGLEW \
        -L$OPTICKS_PREFIX/externals/lib64 -lglfw \
        -lGL  \
        -o $bin

 
        
    # -Wno-unused-private-field \  ## clang-ism ? 

    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
    echo $BASH_SOURCE build DONE
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    dbg__ $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

if [ "${arg/run}" != "$arg" ]; then

    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPTICKS_PREFIX/externals/lib:$OPTICKS_PREFIX/externals/lib64
    echo $BASH_SOURCE : Linux running $bin : with some manual LD_LIBRARY_PATH config 

    [ -z "$DISPLAY" ] && echo $BASH_SOURCE adhoc setting DISPLAY && export DISPLAY=:0 
    $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 3
fi


exit 0 

