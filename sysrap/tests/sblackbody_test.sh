#!/bin/bash

usage(){ cat << EOU

~/o/sysrap/tests/sblackbody_test.sh

EOU
}


name=sblackbody_test

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}


export FOLD=$TMP/$name
bin=$FOLD/$name
script=$name.py

mkdir -p $(dirname $bin)

cd $(dirname $(realpath $BASH_SOURCE))

defarg=info_gcc_run_pdb
arg=${1:-$defarg}

#test=planck_spectral_radiance_set
#test=planck_cdf
#test=planck_icdf
test=planck_sample

export TEST=${TEST:-$test}


vars="BASH_SOURCE arg name FOLD bin script PWD TEST"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/gcc}" != "$arg" ]; then
    gcc $name.cc -g -std=c++17 -lstdc++ -lm -I.. -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : gcc error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

if [ "${arg/dbg}" != "$arg" ]; then
    gdb -ex r $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi


if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : pdb error && exit 3
fi

exit 0


