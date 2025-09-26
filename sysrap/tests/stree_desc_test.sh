#!/bin/bash
usage(){ cat << EOU
stree_desc_test.sh 
=====================

~/o/sysrap/tests/stree_desc_test.sh

EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

#defarg="build_run_ana"
defarg="build_run"
arg=${1:-$defarg}

name=stree_desc_test 
script=$name.py 

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
export FOLD=$TMP/$name
mkdir -p $FOLD
bin=$FOLD/$name 

source $HOME/.opticks/GEOM/GEOM.sh 
export stree_level=1 

CUDA_PREFIX=${CUDA_PREFIX:-/usr/local/cuda}

if [ "${arg/info}" != "$arg" ]; then 
    vars="BASH_SOURCE BASE FOLD"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done
fi 

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc \
          ../sn.cc \
          ../snd.cc \
          ../scsg.cc \
          ../s_pa.cc \
          ../s_tv.cc \
          ../s_bb.cc \
          ../s_csg.cc \
          -g -std=c++17 -lstdc++ \
          -I.. \
          -DWITH_CHILD \
          -I$CUDA_PREFIX/include \
          -I$OPTICKS_PREFIX/externals/glm/glm \
          -lm \
          -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    source dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi 

exit 0 

