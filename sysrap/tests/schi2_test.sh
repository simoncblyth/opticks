#!/bin/bash

usage(){ cat << EOU
schi2_test.sh
===============

Compare different ways to calculate chi2 p_value

EOU
}


name=schi2_test

cd $(dirname $(realpath $BASH_SOURCE))

defarg="info_gcc_run"
arg=${1:-$defarg}

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name
script=$name.py


vv="BASH_SOURCE defarg arg tmp TMP FOLD bin script"

if [[ "$arg" =~ info ]]; then
   for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [[ "$arg" =~ gcc ]]; then
   gcc $name.cc -std=c++17 -lstdc++ -lm -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE gcc error && exit 1
fi

if [[ "$arg" =~ run ]]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

if [[ "$arg" =~ ana ]]; then
   ${IPYTHON:-ipython} $script
   [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
fi

exit 0


