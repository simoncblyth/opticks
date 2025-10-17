#!/bin/bash
usage(){ cat << EOU

~/o/sysrap/tests/SProfTest.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SProfTest

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name

export SProf__WRITE=1

#test=Add_Write_Read
test=ALL

export TEST=${TEST:-$test}

defarg="info_build_run_ls"
arg=${1:-$defarg}

vv="BASH_SOURCE PWD name defarg arg tmp TMP FOLD SProf__WRITE test TEST"

if [ "${arg/info}" != "$arg" ]; then
   for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
   gcc $name.cc ../SProf.cc -std=c++17 -lstdc++ -I.. -g -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE - build error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
   pushd $FOLD
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE - run error && exit 2
   popd
fi

if [ "${arg/ls}" != "$arg" ]; then
   echo ls -alst $FOLD
   ls -alst $FOLD
   cat $FOLD/SProf.txt
fi


exit 0

