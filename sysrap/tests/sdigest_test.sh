#!/bin/bash
usage(){ cat << EOU
sdigest_test.sh
================

~/o/sysrap/tests/sdigest_test.sh


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


defarg=info_build_run
arg=${1:-$defarg}

vars="BASH_SOURCE defarg arg name bin"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then
   gcc $name.cc -std=c++11 -Wall -lstdc++ $opt -I.. -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

exit 0 


