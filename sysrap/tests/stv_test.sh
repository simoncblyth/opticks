#!/bin/bash -l 

name=stv_test 

export FOLD=/tmp/$name 
mkdir -p $FOLD
bin=$FOLD/$name

defarg=build_run
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then
   gcc $name.cc -g -std=c++11 -lstdc++ -I.. -I$OPTICKS_PREFIX/externals/glm/glm -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/dbg}" != "$arg" ]; then
   case $(uname) in
   Darwin) lldb__ $bin ;;
   Linux)  gdb__ $bin ;;
   esac
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi 




exit 0 


