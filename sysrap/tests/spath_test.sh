#!/bin/bash
usage(){ cat << EOU
spath_test.sh
==============

TEST=Resolve3 ~/opticks/sysrap/tests/spath_test.sh

EOU
} 

cd $(dirname $(realpath $BASH_SOURCE))
export SDIR=$PWD

tmp=/tmp/$USER/opticks
export TMP=${TMP:-$tmp}

name=spath_test 
export FOLD=$TMP/$name

bin=$FOLD/$name
mkdir -p $FOLD

test=Filesize
export TEST=${TEST:-$test}

export EXECUTABLE=$bin

source $HOME/.opticks/GEOM/GEOM.sh 

vars="BASH_SOURCE SDIR PWD TMP FOLD name bin GEOM TMP TEST"
for var in $vars ; do printf "%25s : %s\n" "$var" "${!var}" ; done 

gcc $name.cc -g -std=c++17 -lstdc++ -I.. -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1 


pwd
echo $(realpath $PWD)

$bin
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 

exit 0 


