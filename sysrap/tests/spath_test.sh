#!/bin/bash -l 

#TMP=/tmp/somewhere/other/than/here
cd $(dirname $BASH_SOURCE)

name=spath_test 
bin=${TMP:-/tmp/$USER/opticks}/$name
mkdir -p $(dirname $bin)


export EXECUTABLE=$bin

source $HOME/.opticks/GEOM/GEOM.sh 

vars="BASH_SOURCE name bin GEOM TMP"
for var in $vars ; do printf "%25s : %s\n" "$var" "${!var}" ; done 

gcc $name.cc -g -std=c++17 -lstdc++ -I.. -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1 

#unset TMP

#lldb__ $bin
$bin
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 

exit 0 


