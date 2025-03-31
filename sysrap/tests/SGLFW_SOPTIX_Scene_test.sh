#!/bin/bash
usage(){ cat << EOU
SGLFW_SOPTIX_Scene_test.sh : triangulated raytrace and rasterized visualization
=================================================================================

NB because this loads a pre-existing SScene it is necessary to
regenerate the SScene from Geant4 using eg jok-tds when there
are geometry issues.


Assuming the scene folder exists already::

    ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.sh
    SCENE=0 ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.sh
    SCENE=1 ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.sh   ## default
    SCENE=2 ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.sh
    SCENE=3 ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.sh
    ## SCENE picks between different scene directories

As this uses GL interop it may be necessary to select the display GPU::

    CUDA_VISIBLE_DEVICES=1 ~/o/sysrap/tests/ssst.sh


Impl::

    ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.cc


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


#source $HOME/.opticks/GEOM/GEOM.sh
#[ -z "$GEOM" ] && echo $BASH_SOURCE FATAL GEOM $GEOM IS REQUIRTED && exit 1


export FOLD=/tmp/$USER/opticks/$name
bin=$FOLD/$name
mkdir -p $FOLD

export BASE=$TMP/GEOM/$GEOM/$name


if [ -n "$LOG" ]; then
   export SOPTIX_Scene__DUMP=1
   export SGLFW_Scene__DUMP=1
   echo $BASH_SOURCE - LOG defined - enable dumping
else
   echo $BASH_SOURCE - run with LOG defined for dumping
fi


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


source $HOME/.opticks/GEOM/GEOM.sh

scene=0
case ${SCENE:-$scene} in
0) scene_fold=$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim ;;
1) scene_fold=/tmp/SScene_test ;;
2) scene_fold=$TMP/G4CXOpticks_setGeometry_Test/$GEOM/CSGFoundry/SSim ;;
3) scene_fold=$TMP/U4TreeCreateSSimTest/$GEOM/SSim ;;
4) scene_fold=/cvmfs/opticks.ihep.ac.cn/.opticks/GEOM/$GEOM/CSGFoundry/SSim ;;
esac
export SCENE_FOLD=${SCENE_FOLD:-$scene_fold}


shader_fold=../../examples/UseShaderSGLFW_SScene_encapsulated/gl
export SHADER_FOLD=${SHADER_FOLD:-$shader_fold}

dump=0
DUMP=${DUMP:-$dump}
export SGLM__set_frame_DUMP=$DUMP

#export SGLFW_SOPTIX_Scene_test_DUMP=1


#wh=1024,768
wh=2560,1440

#eye=0.1,0,-10
#eye=-1,-1,0
#eye=-10,-10,0
#eye=-10,0,0
#eye=0,-10,0
eye=-1,-1,0
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


#export SOPTIX_SBT__initHitgroup_DUMP=1



defarg="info_ptx_xir_build_run"
arg=${1:-$defarg}


if [ ! -d "$SCENE_FOLD/scene" ]; then
  echo $BASH_SOURCE : FATAL SCENE_FOLD $SCENE_FOLD does not contain scene
  echo $BASH_SOURCE : with newly created CSGFoundry/SSim there is no need for manual SCENE_FOLD as will be in CSGFoundry/SSim/scene
  #arg=info
fi



PATH=$PATH:$CUDA_PREFIX/bin

vars="BASH_SOURCE defarg arg CUDA_PREFIX OPTIX_PREFIX cuda_l SCENE_FOLD FOLD SOPTIX_PTX SOPTIX_XIR SOPTIX_KERNEL bin SGLFW_FRAME GEOM BASE"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done
fi

if [ "${arg/ptx}" != "$arg" ]; then
    echo $BASH_SOURCE ptx
    nvcc $cu \
        -ptx -std=c++11 \
        -c \
        -lineinfo \
        -use_fast_math \
        -I.. \
        -I$CUDA_PREFIX/include  \
        -I$OPTIX_PREFIX/include  \
        -o $ptx
    [ $? -ne 0 ] && echo $BASH_SOURCE : ptx build error && exit 1
    ls -alst $ptx
    echo $BASH_SOURCE ptx DONE
fi


xir-notes(){ cat << EOU

   nvcc warning : '--device-debug (-G)' overrides '--generate-line-info (-lineinfo)'

   With "-G" and default options::

       [ 2][COMPILE FEEDBACK]: COMPILE ERROR: Optimized debugging is not supported.
        Module is built with full debug info, but requested debug level is not
        "OPTIX_COMPILE_DEBUG_LEVEL_FULL".

EOU
}

if [ "${arg/xir}" != "$arg" ]; then
    echo $BASH_SOURCE xir
    nvcc $cu \
        -optix-ir \
        -std=c++11 \
        -c \
        -lineinfo \
        -use_fast_math \
        -I.. \
        -I$CUDA_PREFIX/include  \
        -I$OPTIX_PREFIX/include  \
        -o $xir
    [ $? -ne 0 ] && echo $BASH_SOURCE : xir build error && exit 1
    ls -alst $xir
    echo $BASH_SOURCE xir DONE
fi


gdb__()
{
    : opticks/opticks.bash prepares and invokes gdb - sets up breakpoints based on BP envvar containing space delimited symbols;
    if [ -z "$BP" ]; then
        H="";
        B="";
        T="-ex r";
    else
        H="-ex \"set breakpoint pending on\"";
        B="";
        for bp in $BP;
        do
            B="$B -ex \"break $bp\" ";
        done;
        T="-ex \"info break\" -ex r";
    fi;
    local runline="gdb $H $B $T --args $* ";
    echo $runline;
    date;
    eval $runline;
    date
}



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
        -I$OPTICKS_PREFIX/include/SysRap \
        -I$OPTICKS_PREFIX/externals/glm/glm \
        -I$OPTICKS_PREFIX/externals/include \
        -I$CUDA_PREFIX/include \
        -I$OPTIX_PREFIX/include \
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

if [ "${arg/dbg}" != "$arg" -o -n "$GDB" ]; then
    #dbg__ $bin
    gdb__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

if [ "${arg/run}" != "$arg" ]; then

    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPTICKS_PREFIX/externals/lib:$OPTICKS_PREFIX/externals/lib64
    echo $BASH_SOURCE : Linux running $bin : with some manual LD_LIBRARY_PATH config

    [ -z "$DISPLAY" ] && echo $BASH_SOURCE adhoc setting DISPLAY && export DISPLAY=:0
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

