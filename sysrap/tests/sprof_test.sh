#!/bin/bash -l 
usage(){ cat << EOU
sprof_test.sh 
==============

::

   ~/opticks/sysrap/tests/sprof_test.sh

EOU
}

name=sprof_test 

cd $(dirname $BASH_SOURCE)

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}

bin=$TMP/$name
mkdir -p $(dirname $bin)

gcc $name.cc -I.. -std=c++11 -lstdc++ -lm -o $bin && $bin


