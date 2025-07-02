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

bin=$FOLD/$name
script=$name.py

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}
opt=-DMOCK_CUDA

#defarg="info_build_run_ana"
defarg="info_build_run_pdb"
arg=${1:-$defarg}
vars="BASH_SOURCE name arg FOLD bin script opt"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc $opt -std=c++17 -lstdc++ -lm -I.. -I$CUDA_PREFIX/include -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1
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

