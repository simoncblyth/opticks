#!/bin/bash
usage(){ cat << EOU
SGLFW_SOPTIX_Scene_test.sh : triangulated raytrace and rasterized visualization
=================================================================================


Assuming the scene folder exists already::

    ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.sh

As this uses GL interop it may be necessary to select the display GPU::

    CUDA_VISIBLE_DEVICES=1 ~/o/sysrap/tests/ssst.sh

Impl::

    ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.cc


TODO : Incorporate this into release
--------------------------------------


One step way to create the stree and scene from gdml or other geometry source
---------------------------------------------------------------------------------

::

    ~/o/u4/tests/U4TreeCreateSSimTest.sh


Two step way to create the scene folder from gdml or other geometry source
-----------------------------------------------------------------------------

1. load geometry and use U4Tree::Create to convert to stree.h and persist the tree::

    ~/o/u4/tests/U4TreeCreateTest.sh
        # create and persist stree.h (eg from gdml or j/PMTSim via GEOM config)

2. load stree and create corresponding SScene::

    ~/o/sysrap/tests/SScene_test.sh
        # create and persist SScene.h from loaded stree.h


Simpler Prior Developments
---------------------------

Related simpler priors::

    ~/o/sysrap/tests/SOPTIX_Scene_test.sh
    ~/o/sysrap/tests/SOPTIX_Scene_test.cc



Which GL/glew.h is picked up ?
---------------------------------

On changing env from JUNO-opticks to ONLY-opticks build directly with
opticks-full note that have changed the OpenGL version in use
causing missing symbol GL_CONTEXT_LOST

Fixed that with GL_VERSION_4_5 check, but why the older version ?

* probably just a change in glew header not a change in actual GL version used

Adding "-M" to gcc commandline lists all the
included headers in Makefile dependency format.
This shows are picking up system headers::

    /usr/include/GL/glew.h
    /usr/include/GL/glu.h
    /usr/include/GL/gl.h
    /usr/include/GLFW/glfw3.h

Which corresponds to::

    //SGLFW::init GL_RENDERER [NVIDIA TITAN RTX/PCIe/SSE2]
    //SGLFW::init GL_VERSION [4.1.0 NVIDIA 515.43.04]

HMM: this maybe because I removed glew and glfw from the
standard externals ?

HMM: but after installing those get::

   undefined symbol: glfwSetWindowMaximizeCallback

when using::

    /data/blyth/opticks_Debug/externals/include/GL/glew.h
    /usr/include/GL/glu.h
    /usr/include/GL/gl.h
    /data/blyth/opticks_Debug/externals/include/GLFW/glfw3.h

Return to system with::

   glfw-
   glfw-manifest-wipe
   glfw-manifest-wipe | sh

   glew-
   glew-manifest-wipe
   glew-manifest-wipe | sh



EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
name=SGLFW_SOPTIX_Scene_test


source $HOME/.opticks/GEOM/GEOM.sh
[ -z "$GEOM" ] && echo $BASH_SOURCE FATAL GEOM $GEOM IS REQUIRTED && exit 1

_CFB=${GEOM}_CFBaseFromGEOM

if [ ! -d "${!_CFB}/CSGFoundry/SSim/scene" ]; then
   echo $BASH_SOURCE : FATAL GEOM $GEOM ${_CFB} ${!_CFB}
   exit 1
fi

source $HOME/.opticks/GEOM/EVT.sh   ##  optionally sets AFOLD BFOLD where event info is loaded from
source $HOME/.opticks/GEOM/MOI.sh   ## optionally sets MOI envvar controlling initial viewpoint


export FOLD=/tmp/$USER/opticks/$name
bin=$FOLD/$name
mkdir -p $FOLD

export BASE=$TMP/GEOM/$GEOM/$name



logging()
{
   type $FUNCNAME
   #export SGLFW_Scene__DUMP=1
   #export SGLM__set_frame_DUMP=1
   #export SGLFW_SOPTIX_Scene_test_DUMP=1
   #export SOPTIX_SBT__initHitgroup_DUMP=1
   export SOPTIX_Options__LEVEL=1
   export SOPTIX_Scene__DUMP=1
}
[ -n "$LOG" ] && logging



cu=../SOPTIX.cu
ptx=$FOLD/SOPTIX.ptx
xir=$FOLD/SOPTIX.optixir

export SOPTIX_PTX=$ptx
export SOPTIX_XIR=$xir
export SOPTIX_KERNEL=$SOPTIX_PTX
#export SOPTIX_KERNEL=$SOPTIX_XIR

export SGLFW__DEPTH=1



# when using CMake generated ptx will be smth like:$OPTICKS_PREFIX/ptx/sysrap_generated_SOPTIX.cu.ptx
# following pattern $OPTICKS_PREFIX/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx"

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}
for l in lib lib64 ; do [ -d "$CUDA_PREFIX/$l" ] && cuda_l=$l ; done

optix_prefix=${OPTICKS_OPTIX_PREFIX}
OPTIX_PREFIX=${OPTIX_PREFIX:-$optix_prefix}
[ -z "$OPTIX_PREFIX" ] && echo $BASH_SOURCE - ERROR no OPTIX_PREFIX or OPTICKS_OPTIX_PREFIX && exit 1

sysrap_dir=..
SYSRAP_DIR=${SYSRAP_DIR:-$sysrap_dir}





#wh=1024,768
wh=2560,1440

#eye=0.1,0,-10
#eye=-1,-1,0
#eye=-10,-10,0
#eye=-10,0,0
#eye=0,-10,0
#eye=-1,-1,0
eye=1,0,0
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


handle=-1 # -1:IAS 0...8 GAS indices
export HANDLE=${HANDLE:-$handle}

frame=-1
export SGLFW_FRAME=${SGLFW_FRAME:-$frame}

vizmask=t   # 0xff no masking
#vizmask=t0  # 0xfe mask global
export VIZMASK=${VIZMASK:-$vizmask}





defarg="info_ptx_xir_build_run"
arg=${1:-$defarg}




PATH=$PATH:$CUDA_PREFIX/bin

vars="BASH_SOURCE defarg arg CUDA_PREFIX OPTIX_PREFIX cuda_l SCENE_FOLD FOLD SOPTIX_PTX SOPTIX_XIR SOPTIX_KERNEL bin SGLFW_FRAME GEOM BASE"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done
fi

if [ "${arg/ptx}" != "$arg" -o "${arg/xir}" != "$arg" ]; then

    opt=""
    if [ "${arg/ptx}" != "$arg" ]; then
        out=$ptx
        opt=-ptx
    elif [ "${arg/xir}" != "$arg" ]; then
        out=$xir
        opt=-optix-ir
    fi
    echo $BASH_SOURCE arg $arg out $out opt $opt

    # -DDBG_PIDX

    nvcc $cu \
        $opt \
        -std=c++11 \
        -c \
        -lineinfo \
        -use_fast_math \
        -I.. \
        -I$CUDA_PREFIX/include  \
        -I$OPTIX_PREFIX/include  \
        -o $out
    [ $? -ne 0 ] && echo $BASH_SOURCE : out $out build error && exit 1
    ls -alst $out

    echo $BASH_SOURCE out $out DONE
fi


xir-notes(){ cat << EOU

   nvcc warning : '--device-debug (-G)' overrides '--generate-line-info (-lineinfo)'

   With "-G" and default options::

       [ 2][COMPILE FEEDBACK]: COMPILE ERROR: Optimized debugging is not supported.
        Module is built with full debug info, but requested debug level is not
        "OPTIX_COMPILE_DEBUG_LEVEL_FULL".

EOU
}



if [ "${arg/build}" != "$arg" ]; then

    echo $BASH_SOURCE build
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
        -I$OPTICKS_PREFIX/include/SysRap \
        -I$OPTICKS_PREFIX/externals/glm/glm \
        -I$OPTICKS_PREFIX/externals/include \
        -I$CUDA_PREFIX/include \
        -I$OPTIX_PREFIX/include \
        -L$CUDA_PREFIX/$cuda_l -lcudart \
        -lstdc++ \
        -lm -ldl \
        -L$OPTICKS_PREFIX/lib64 -lSysRap \
        -L$OPTICKS_PREFIX/externals/lib -lGLEW \
        -L$OPTICKS_PREFIX/externals/lib64 -lglfw \
        -lGL  \
        -o $bin

    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1
    echo $BASH_SOURCE build DONE
fi

if [ "${arg/dbg}" != "$arg" -o -n "$GDB" ]; then
    source dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 3
fi

if [ "${arg/grab}" != "$arg" ]; then
    source $OPTICKS_HOME/bin/BASE_grab.sh $arg
fi

if [ "${arg/list}" != "$arg" -o "${arg/pub}" != "$arg" ]; then
    source $OPTICKS_HOME/bin/BASE_grab.sh $arg
fi

exit 0

