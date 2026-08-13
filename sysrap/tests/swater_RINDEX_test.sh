#!/bin/bash
usage(){ cat << EOU

~/o/sysrap/tests/swater_RINDEX_test.sh
EOU
}


name=swater_RINDEX_test

cd $(dirname $(realpath $BASH_SOURCE))

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
export FOLD=$TMP/$name

mkdir -p $FOLD

bin=$FOLD/$name
script=$name.py


defarg=info_gcc_run_pdb
arg=${1:-$defarg}

vv="BASH_SOURCE PWD tmp TMP FOLD bin script defarg arg"

if [[ "$arg" =~ info ]]; then
   for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [[ "$arg" =~ gcc ]]; then
   gcc $name.cc -std=c++17 -lstdc++ -lm -I.. -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE - gcc ERROR && exit 1
fi

if [[ "$arg" =~ run ]]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE - run ERROR && exit 2
fi

if [[ "$arg" =~ pdb ]]; then
   ${IPYTHON:-ipython} -i --pdb $script
   [ $? -ne 0 ] && echo $BASH_SOURCE - pdb ERROR && exit 3
fi

exit 0



