#!/bin/bash -l
usage(){ cat << EOU

::

   ~/o/examples/UseOptiX7_CLI/go.sh 
   ~/o/examples/UseOptiX7_CLI/UseOptiX7_CLI.cc 

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=UseOptiX7_CLI
export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}
for l in lib lib64 ; do [ -d "$CUDA_PREFIX/$l" ] && cuda_l=$l ; done

optix_prefix=${OPTICKS_OPTIX_PREFIX}
OPTIX_PREFIX=${OPTIX_PREFIX:-$optix_prefix}

sysrap_dir=$OPTICKS_HOME/sysrap
SYSRAP_DIR=${SYSRAP_DIR:-$sysrap_dir}


vars="BASH_SOURCE name FOLD bin CUDA_PREFIX cuda_l OPTIX_PREFIX SYSRAP_DIR"
for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done

gcc $name.cc \
     -std=c++11 -lstdc++ -lm -ldl -g \
     -I${SYSRAP_DIR} \
     -I${CUDA_PREFIX}/include \
     -I${OPTIX_PREFIX}/include \
     -L${CUDA_PREFIX}/$cuda_l -lcudart \
     -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 

$bin
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 

exit 0 

