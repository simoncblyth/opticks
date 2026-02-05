#!/bin/bash

usage(){ cat << EOU

~/o/sysrap/tests/sfr_test.sh
~/o/sysrap/tests/sfr_test.cc
~/o/sysrap/tests/sfr_test.py

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=sfr_test

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name
script=$name.py

defarg=info_build_run_ana
arg=${1:-$defarg}


vars="BASH_SOURCE name"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%30s : %s\n" "$var" "${!var}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc -std=c++17 -lstdc++ -lm -g -I.. -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1
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


