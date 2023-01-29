#!/bin/bash -l 
usage(){ cat << EOU
stree_load_test.sh 
=====================

See:

* notes/issues/U4Tree_stree_snd_scsg_FAIL_consistent_parent.rst


EOU
}

defarg="build_run_ana"
arg=${1:-$defarg}

name=stree_load_test 
bin=/tmp/$name/$name 

export BASE=/tmp/$USER/opticks/U4TreeCreateTest 
export stree_level=1 
export FOLD=$BASE/stree

if [ ! -d "$BASE/stree" ]; then
    echo $BASH_SOURCE : BASE $BASE
    echo $BASH_SOURCE : BASE directory MUST contain an stree directory : THIS DOES NOT 
    exit 1
fi 

if [ "${arg/info}" != "$arg" ]; then 
    vars="BASH_SOURCE BASE FOLD"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done
fi 

if [ "${arg/build}" != "$arg" ]; then 
    mkdir -p $(dirname $bin)
    gcc $name.cc ../snd.cc ../scsg.cc  \
          -g -std=c++11 -lstdc++ \
          -I.. \
          -I/usr/local/cuda/include \
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
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi 

if [ "${arg/csg}" != "$arg" ]; then 
    FOLD=$FOLD/csg ${IPYTHON:-ipython} --pdb -i ${name}_csg.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi 





exit 0 

