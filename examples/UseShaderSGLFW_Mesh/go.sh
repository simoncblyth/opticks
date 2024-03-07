#!/bin/bash -l
usage(){ cat << EOU
examples/UseShaderSGLFW_Mesh/go.sh
=====================================

Pops up an OpenGL window with a colorful single triangle::

    ~/o/examples/UseShaderSGLFW_Mesh/go.sh 
    ~/o/examples/UseShaderSGLFW_Mesh/go.sh info
    ~/o/examples/UseShaderSGLFW_Mesh/go.sh run

Issues of view matrices
-------------------------

* G4Orb/U4Mesh (CE 0,0,0,100) only becomes visible at EYE=0.1,0,-10 (not 0.1,0,-9) with default TMIN=0.1
* TMIN is effecting OpenGL apparent object size (which is wrong, it should just change frustum near)

* TODO: interactive sliders to ease debugging of these issues, start with just key callbacks (ImGui sliders can come later)
* TODO: numerically establish equivalence of ray trace and rasterized renders, and do that in practice

EOU
}


opticks-
oe-
om-

path=$(realpath $BASH_SOURCE)
sdir=$(dirname $path)
name=$(basename $sdir)

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}
export CUDA_PREFIX

bdir=/tmp/$USER/opticks/$name/build 

mesh_fold=/tmp/U4Mesh_test
export MESH_FOLD=${MESH_FOLD:-$mesh_fold}

shader=wireframe
SHADER=${SHADER:-$shader}
export SHADER_FOLD=$sdir/gl/$SHADER


wh=1024,768
eye=0.1,0,-10
tmin=0.1      


export WH=${WH:-$wh}
export EYE=${EYE:-$eye}
export TMIN=${TMIN:-$tmin}

vars="BASH_SOURCE bdir SHADER_FOLD MESH_FOLD WH EYE"

defarg="info_build_run"
arg=${1:-$defarg}
if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 
    om-cmake $sdir 
    make
    [ $? -ne 0 ] && echo $BASH_SOURCE : make error && exit 1 
    make install   
    [ $? -ne 0 ] && echo $BASH_SOURCE : install error && exit 2 
fi

if [ "${arg/run}" != "$arg" ]; then 
    echo executing $name
    $name
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 3
fi

exit 0 


