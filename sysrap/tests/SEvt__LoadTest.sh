#!/bin/bash -l 
usage(){ cat << EOU
SEvt__LoadTest.sh
====================

TODO: avoid the need for the kitchensink just to load an SEvt 

EOU
}

name=SEvt__LoadTest

export FOLD=/tmp/$name
mkdir -p $FOLD

bin=$FOLD/$name
defarg="info_build_run"
arg=${1:-$defarg}

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

source $HOME/.opticks/GEOM/GEOM.sh 
export BASE=/tmp/$USER/opticks/GEOM/$GEOM/ntds3/ALL1
export AFOLD=$BASE/p001
export BFOLD=$BASE/n001

vars="BASH_SOURCE GEOM FOLD OPTICKS_PREFIX CUDA_PREFIX AFOLD BFOLD"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc \
         ../SEvt.cc \
         ../SEventConfig.cc \
         ../SFrameGenstep.cc \
         ../SOpticksResource.cc \
         ../SGeo.cc \
         ../SOpticksKey.cc \
         ../OpticksPhoton.cc \
         ../SBit.cc \
         ../SAr.cc \
         ../SStr.cc \
         ../SSys.cc \
         ../SGenstep.cc \
         ../SLOG.cc \
         ../SPath.cc \
         ../SEvent.cc \
         ../SDigest.cc \
         ../SProc.cc \
         ../SASCII.cc \
         ../../okconf/OKConf.cc \
         -I.. \
         -I$OPTICKS_PREFIX/include/OKConf \
         -I$OPTICKS_PREFIX/externals/plog/include \
         -I$OPTICKS_PREFIX/externals/glm/glm \
         -I$CUDA_PREFIX/include \
         -std=c++11 -lstdc++ \
         -o $bin  
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin  
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi


exit 0 

