#!/bin/bash -l 
usage(){ cat << EOU

~/o/examples/UseShaderSGLFW_SScene/run.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=UseShaderSGLFW_SScene_encapsulated
bin=$(which $name)

scene_fold=/tmp/SScene_test

#wh=1024,768
wh=2048,1536

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


export SCENE_FOLD=${SCENE_FOLD:-$scene_fold}
export SHADER_FOLD=gl
export WH=${WH:-$wh}
export EYE=${EYE:-$eye}
export LOOK=${LOOK:-$look}
export UP=${UP:-$up}
export TMIN=${TMIN:-$tmin}
export ESCALE=${ESCALE:-$escale}
export CAM=${CAM:-$cam}


$name

case $(uname) in 
  Darwin) otool -L $bin ;;
  Linux)  ldd $bin ;;
esac



