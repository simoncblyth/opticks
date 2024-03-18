#!/bin/bash -l
usage(){ cat << EOU
examples/UseShaderSGLFW_MeshMesh_Instanced/go.sh
====================================================

Started from examples/UseShaderSGLFW_MeshMesh. 
Are aiming to add instancing with an array of transforms
as input separate from SMesh::

    ~/o/examples/UseShaderSGLFW_MeshMesh_Instanced/go.sh 
    ~/o/examples/UseShaderSGLFW_MeshMesh_Instanced/go.sh info
    ~/o/examples/UseShaderSGLFW_MeshMesh_Instanced/go.sh run

Impl::

    ~/o/examples/UseShaderSGLFW_MeshMesh_Instanced/UseShaderSGLFW_MeshMesh_Instanced.cc

Issues of view matrices : TODO: Debug once can flip between raytrace and raster renders
-----------------------------------------------------------------------------------------

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

wh=1024,768

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
export SHADER_FOLD=$sdir/gl
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

if [ "${arg/dbg}" != "$arg" ]; then 
    echo executing dbg__ $name
    dbg__ $name
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 4
fi


exit 0 


