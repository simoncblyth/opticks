#!/bin/bash 
usage(){ cat << EOU
SGenstep_test.sh
=================

~/o/sysrap/tests/SGenstep_test.sh

EOU
}
cd $(dirname $(realpath $BASH_SOURCE))


name=SGenstep_test
export FOLD=/tmp/$name
mkdir -p $FOLD

bin=$FOLD/$name

defarg=info_build_run
arg=${1:-$defarg}

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

#test=Slices_3
test=ALL
export TEST=${TEST:-$test}

vars="BASH_SOURCE name test TEST"

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -std=c++11 -lstdc++ -g -I$CUDA_PREFIX/include -I$OPTICKS_PREFIX/externals/glm/glm -I.. -lm -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

exit 0

