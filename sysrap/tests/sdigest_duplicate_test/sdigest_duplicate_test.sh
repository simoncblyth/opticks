#!/bin/bash
usage(){ cat << EOU
sdigest_duplicate_test.sh
===============================

~/o/sysrap/tests/sdigest_duplicate_test/sdigest_duplicate_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=sdigest_duplicate_test 

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}

FOLD=$TMP/$name
mkdir -p $FOLD

nproc_default=$(nproc)
export NPROC=${NPROC:-$nproc_default}

bin=$FOLD/$name

opt="-Wdeprecated-declarations"
case $(uname) in 
  Darwin) opt="" ;;
   Linux) opt="-lssl -lcrypto " ;;
esac

export HITFOLD=/data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvlarge_evt/A000

defarg=info_build_run
arg=${1:-$defarg}

vars="BASH_SOURCE defarg arg name bin FOLD HITFOLD nproc_default NPROC"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then
   gcc $name.cc -std=c++17 -g -Wall -lstdc++ $opt -I../.. -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi


exit 0 


