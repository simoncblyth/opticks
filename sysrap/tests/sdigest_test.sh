#!/bin/bash
usage(){ cat << EOU
sdigest_test.sh
================

~/o/sysrap/tests/sdigest_test.sh
~/o/sysrap/tests/sdigest_test.sh info_build_hit


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=sdigest_test 
bin=/tmp/$name

opt="-Wdeprecated-declarations"
case $(uname) in 
  Darwin) opt="" ;;
   Linux) opt="-lssl -lcrypto " ;;
esac


opt="$opt -O2 -D_FORTIFY_SOURCE=2 -Wstringop-overflow"


defarg=info_build_run
arg=${1:-$defarg}

vars="BASH_SOURCE defarg arg name bin opt"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then
   gcc $name.cc -std=c++17 -Wall -lstdc++ $opt -I.. -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi


if [ "${arg/hit}" != "$arg" ]; then

   export HITFOLD=/data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_vvvlarge_evt/A000
   TEST=Hit $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE hit error && exit 2
fi




exit 0 


