#!/bin/bash -l 

name=SBnd_test 
export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

source $HOME/.opticks/GEOM/GEOM.sh 


defarg="build_run_ana"
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then 
   gcc $name.cc -I.. -g -std=c++11 -lstdc++ -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
   case $(uname) in 
     Darwin) lldb__ $bin ;; 
     Linux)  gdb__ $bin ;;
   esac
   [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 3
fi 

if [ "${arg/ana}" != "$arg" ]; then 
   ${IPYTHON:-ipython} --pdb -i $name.py 
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi 





exit 0 

