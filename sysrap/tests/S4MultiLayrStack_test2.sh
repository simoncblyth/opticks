#!/bin/bash
usage(){ cat << EOU

~/o/sysrap/tests/S4MultiLayrStack_test2.sh

EOU
}


name=S4MultiLayrStack_test2

cd $(dirname $(realpath $BASH_SOURCE))

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
export FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name
script=

defarg="build_run_ana"
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then
   gcc $name.cc -std=c++17 -lstdc++ -I$HOME/np -DWITH_SCOMPLEX  -I.. -o $bin  
   [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then
   $bin 
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then
   ${IPYTHON:-ipython} --pdb -i $name.py 
   [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3 
fi 

exit 0 


