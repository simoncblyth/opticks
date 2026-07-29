#/bin/bash
usage(){ cat << EOU
smath_test.sh
==============

~/o/sysrap/tests/smath_test.sh

EOU
}

name=smath_test

cd $(dirname $(realpath $BASH_SOURCE))


tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
export FOLD=$TMP/$name
mkdir -p $FOLD

cuobj=$FOLD/${name}_cu.obj
ccobj=$FOLD/${name}_cc.obj
bin=$FOLD/$name
script=$name.py

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

#opt=-DMOCK_CUDA


test=log_cu_float
export TEST=${TEST:-$test}

#defarg="info_gcc_nvcc_link_run_ana"
defarg="info_gcc_nvcc_link_run_pdb"
arg=${1:-$defarg}
vars="BASH_SOURCE name arg FOLD bin script opt test TEST"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/gcc}" != "$arg" ]; then
    gcc -c $name.cc $opt -std=c++17 -g -lstdc++ -lm -I.. -I$CUDA_PREFIX/include -o $ccobj
    [ $? -ne 0 ] && echo $BASH_SOURCE : gcc error && exit 1
fi

if [ "${arg/nvcc}" != "$arg" ]; then
    nvcc -c $name.cu $opt --std=c++17 -I.. -o $cuobj
    [ $? -ne 0 ] && echo $BASH_SOURCE : nvcc error && exit 1
fi

if [ "${arg/link}" != "$arg" ]; then
    gcc $ccobj $cuobj -o $bin -lstdc++ -L/usr/local/cuda/lib64 -lcudart -lm
    [ $? -ne 0 ] && echo $BASH_SOURCE : link error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi

exit 0

