#!/bin/bash -l 
usage(){ cat << EOU
stree_loadsave_test.sh 
========================

EOU
}


SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)

defarg="build_run_ana"
arg=${1:-$defarg}

name=stree_loadsave_test 
export FOLD=/tmp/$name
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
    gcc $SDIR/$name.cc $SDIR/../snd.cc $SDIR/../scsg.cc  \
          -g -std=c++11 -lstdc++ \
          -I$SDIR/.. \
          -I$CUDA_PREFIX/include \
          -I$OPTICKS_PREFIX/externals/glm/glm \
          -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    case $(uname) in
       Darwin) lldb__ $bin ;;
       Linux)  gdb__  $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $SDIR/$name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi 


exit 0 

