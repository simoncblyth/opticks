#!/bin/bash -l
usage(){ cat << EOU
tree_test.sh
=============

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
name=tree_test
bin=/tmp/$name 

defarg="info_build_run"
arg=${1:-$defarg}
vars="BASH_SOURCE SDIR arg name bin"

#export VERBOSE=1

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then
    gcc $SDIR/$name.cc -g -I$SDIR/.. -std=c++11 -lstdc++ -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : build fail && exit 1
fi 

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run fail && exit 2
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    case $(uname) in
       Darwin) lldb__ $bin ;;
       Linux)  gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg fail && exit 3
fi 

exit 0 
 
