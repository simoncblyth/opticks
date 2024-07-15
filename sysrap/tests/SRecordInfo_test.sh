#!/bin/bash  
usage(){ cat << EOU
SRecordInfo_test.sh
===============



EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SRecordInfo_test
export SRECORDER_FOLD=/tmp/$name
mkdir -p $SRECORDER_FOLD 
bin=$SRECORDER_FOLD/$name

export SRECORD_PATH=/tmp/sphoton_test/record.npy



vars="BASH_SOURCE PWD SRECORDER_FOLD bin "

defarg=info_build_run
arg=${1:-$defarg}

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

glm_prefix=$OPTICKS_PREFIX/externals/glm/glm
GLM_PREFIX=${GLM_PREFIX:-$glm_prefix}



if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi   

if [ "${arg/build}" != "$arg" ]; then 

    echo $BASH_SOURCE [ building 
    gcc  $name.cc \
         -std=c++11 -lstdc++ -lm \
         -I.. -g \
         -I$OPTICKS_PREFIX/externals/glm/glm \
         -I$OPTICKS_PREFIX/externals/include \
         -I$OPTICKS_PREFIX/include/SysRap \
         -I$CUDA_PREFIX/include \
         -lstdc++ \
         -lm \
         -L$OPTICKS_PREFIX/externals/lib -lGLEW \
         -L$OPTICKS_PREFIX/externals/lib64 -lglfw \
         -I$GLM_PREFIX \
         -I${CUDA_PREFIX}/include \
         -o $bin

    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1
    echo $BASH_SOURCE ] building 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi 

exit 0 
