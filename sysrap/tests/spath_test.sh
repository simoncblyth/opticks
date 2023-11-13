#!/bin/bash -l 

#TMP=/tmp/somewhere/other/than/here

name=spath_test 
bin=${TMP:-/tmp/$USER/opticks}/$name
mkdir -p $(dirname $bin)

gcc $name.cc -g -std=c++11 -lstdc++ -I.. -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1 

#unset TMP

#lldb__ $bin
$bin
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 

exit 0 


