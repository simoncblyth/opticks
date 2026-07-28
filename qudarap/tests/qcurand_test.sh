#!/bin/bash

usage(){ cat << EOU

~/o/qudarap/tests/qcurand_test.sh

EOU
}

defarg="info_nvcc_gcc_link_run_pdb"
arg=${1:-$defarg}

name=qcurand_test

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
export FOLD=$TMP/$name
mkdir -p $FOLD

cd $(dirname $(realpath $BASH_SOURCE))

bin=$FOLD/$name
script=$name.py


vv="BASH_SOURCE defarg arg name tmp TMP FOLD PWD bin"

if [[ "$arg" =~ "info" ]]; then
   for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [[ "$arg" =~ "nvcc" ]]; then
   nvcc -c $name.cu -I.. --std=c++17 -o $FOLD/$name.o -O3
   [ $? -ne 0 ] && echo $BASH_SOURCE - nvcc ERROR for $name.cu && exit 1
fi

if [[ "$arg" =~ "gcc" ]]; then
   gcc -c $name.cc -I.. -I../../sysrap -std=c++17 -o $FOLD/${name}_host.o -O3
   [ $? -ne 0 ] && echo $BASH_SOURCE - nvcc ERROR for $name.cu && exit 1
fi

if [[ "$arg" =~ "link" ]]; then
   gcc $FOLD/${name}_host.o $FOLD/$name.o -o $bin -lstdc++ -L/usr/local/cuda/lib64 -lcudart -lcurand
   [ $? -ne 0 ] && echo $BASH_SOURCE - link ERROR for bin $bin && exit 1
fi

if [[ "$arg" =~ "run" ]]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE - run ERROR for bin $bin && exit 1
fi

if [[ "$arg" =~ "pdb" ]]; then
   ${IPYTHON:-ipython} --pdb -i $script
   [ $? -ne 0 ] && echo $BASH_SOURCE - pdb ERROR for script $script && exit 1
fi


exit 0

