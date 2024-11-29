#!/bin/bash

usage(){ cat << EOU
spath_test.sh
==============

~/opticks/sysrap/tests/spath_test.sh

EOU
} 

cd $(dirname $BASH_SOURCE)
export SDIR=$PWD

name=spath_test 
bin=${TMP:-/tmp/$USER/opticks}/$name
mkdir -p $(dirname $bin)

test=Filesize
export TEST=${TEST:-$test}


export EXECUTABLE=$bin

source $HOME/.opticks/GEOM/GEOM.sh 

vars="BASH_SOURCE SDIR PWD name bin GEOM TMP TEST"
for var in $vars ; do printf "%25s : %s\n" "$var" "${!var}" ; done 

gcc $name.cc -g -std=c++17 -lstdc++ -I.. -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1 

#unset TMP

pwd
echo $(realpath $PWD)


$bin
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 

exit 0 


