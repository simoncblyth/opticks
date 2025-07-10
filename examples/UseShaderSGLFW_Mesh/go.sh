#!/bin/bash
usage(){ cat << EOU
examples/UseShaderSGLFW_Mesh/go.sh
=====================================

Pops up an OpenGL window with a colorful visualization of a mesh of triangles::

    ~/o/examples/UseShaderSGLFW_Mesh/go.sh
    ~/o/examples/UseShaderSGLFW_Mesh/go.sh info
    ~/o/examples/UseShaderSGLFW_Mesh/go.sh run

Impl::

    ~/o/examples/UseShaderSGLFW_Mesh/UseShaderSGLFW_Mesh.cc


Issues of view matrices
-------------------------

* G4Orb/U4Mesh (CE 0,0,0,100) only becomes visible at EYE=0.1,0,-10 (not 0.1,0,-9) with default TMIN=0.1
* TMIN is effecting OpenGL apparent object size (which is wrong, it should just change frustum near)

* TODO: interactive sliders to ease debugging of these issues, start with just key callbacks (ImGui sliders can come later)
* TODO: numerically establish equivalence of ray trace and rasterized renders, and do that in practice

EOU
}


path=$(realpath $BASH_SOURCE)
sdir=$(dirname $path)
name=$(basename $sdir)

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}
export CUDA_PREFIX

bdir=/tmp/$USER/opticks/$name/build
idir=/tmp/$USER/opticks/$name/install


PREFIX=$idir
bin=$PREFIX/lib/$name



#solid=Orb
#solid=Torus   # HUH: appears like Orb
solid=Box
#solid=Tet
SOLID=${SOLID:-$solid}
mesh_fold=/tmp/U4Mesh_test/$SOLID

shader=wireframe
#shader=normal
SHADER=${SHADER:-$shader}


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

#cam=perspective
cam=orthographic

tmin=0.1
#escale=asis
escale=extent

export MESH_FOLD=${MESH_FOLD:-$mesh_fold}
export SHADER_FOLD=$sdir/gl/$SHADER
export WH=${WH:-$wh}
export EYE=${EYE:-$eye}
export LOOK=${LOOK:-$look}
export UP=${UP:-$up}
export TMIN=${TMIN:-$tmin}
export ESCALE=${ESCALE:-$escale}
export CAM=${CAM:-$cam}



vars="BASH_SOURCE bdir SHADER_FOLD MESH_FOLD WH EYE"

defarg="info_build_run"
arg=${1:-$defarg}
if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
    rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd

    cmake $sdir \
       -DCMAKE_BUILD_TYPE=Debug \
       -DOPTICKS_PREFIX=$OPTICKS_PREFIX \
       -DCMAKE_INSTALL_PREFIX=$PREFIX \
       -DCMAKE_MODULE_PATH=$OPTICKS_PREFIX/cmake/Modules


    make
    [ $? -ne 0 ] && echo $BASH_SOURCE : make error && exit 1
    make install
    [ $? -ne 0 ] && echo $BASH_SOURCE : install error && exit 2
fi

if [ "${arg/run}" != "$arg" ]; then
    echo executing $bin
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 3
fi

if [ "${arg/dbg}" != "$arg" ]; then
    echo executing dbg__ $bin
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 4
fi




exit 0


