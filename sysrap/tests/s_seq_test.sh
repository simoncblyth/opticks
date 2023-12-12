#!/bin/bash -l 
usage(){ cat << EOU
s_seq_test.sh
==============

::

   ~/o/sysrap/tests/s_seq_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=s_seq_test 

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
bin=$TMP/$name 

[ -n "$LARGE" ] && echo $BASH_SOURCE ENABLE LARGE && export s_seq__SeqPath_DEFAULT_LARGE=1 
env | grep s_seq

gcc $name.cc -I.. -std=c++11 -lstdc++ -o $bin 
[ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 

$bin
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 

exit 0 

