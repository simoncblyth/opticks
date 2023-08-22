#!/bin/bash -l 

name=stra_test 
export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

defarg="info_build_run" 
arg=${1:-$defarg}
opt=-g

vars="BASH_SOURCE arg opt FOLD"

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
   gcc $name.cc $opt -std=c++11 -lstdc++ -I.. -I$OPTICKS_PREFIX/externals/glm/glm -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
   $bin 
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
   case $(uname) in
      Darwin) lldb__ $bin  ;;
      Linux)   gdb__ $bin ;;
   esac
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi 

exit 0 









