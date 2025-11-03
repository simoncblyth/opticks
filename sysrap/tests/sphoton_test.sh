#!/bin/bash
usage(){ cat << EOU
sphoton_test.sh
===============

::

   ~/o/sysrap/tests/sphoton_test.sh
   TEST=make_record_array ~/o/sysrap/tests/sphoton_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=sphoton_test

#defarg="info_build_run_ana"
defarg="info_build_run_pdb"
arg=${1:-$defarg}

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name
script=$name.py

vars="BASH_SOURCE PWD FOLD CUDA_PREFIX name bin script TEST"

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

#test=make_record_array
#test=set_flag
#test=add_flagmask
#test=ChangeTimeInsitu
#test=index
test=demoarray

TEST=${TEST:-$test}
export TEST

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc -std=c++17 -lstdc++ -lm -lcrypto -lssl \
           -I.. \
           -I$CUDA_PREFIX/include \
           -I$OPTICKS_PREFIX/externals/glm/glm \
           -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE pdb error && exit 3
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi

exit 0

