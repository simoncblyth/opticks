#!/usr/bin/env bash

usage(){ cat << EOU

~/o/sysrap/tests/sfilesystem_test.sh


EOU
}

name=sfilesystem_test

defarg=info_gcc_run
arg=${1:-$defarg}

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
FOLD=$TMP/$name

mkdir -p $FOLD
bin=$FOLD/$name

cd $(dirname $(realpath $BASH_SOURCE))

vv="BASH_SOURCE PWD name defarg arg tmp TMP FOLD bin"

if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/gcc}" != "$arg" ]; then
   gcc $name.cc -std=c++17 -lstdc++ -I.. -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE - gcc error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then

   cd $FOLD

   rm -rf container_dir
   mkdir container_dir
   mkdir container_dir/sreport_0000200
   mkdir container_dir/sreport_0000214

   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE - run error && exit 1
fi

exit 0


