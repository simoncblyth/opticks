#!/bin/bash -l 
usage(){ cat << EOU
sndtree_test.sh 
=====================


EOU
}


SDIR=$(dirname $BASH_SOURCE)

#defarg="build_run_ana"
defarg="build_run" 

arg=${1:-$defarg}

name=sndtree_test 
bin=/tmp/$name/$name 

if [ "${arg/build}" != "$arg" ]; then 
    mkdir -p $(dirname $bin)
    echo build
    gcc $SDIR/$name.cc $SDIR/../snd.cc $SDIR/../scsg.cc  \
          -g -std=c++11 -lstdc++ \
          -I$SDIR/.. \
          -I/usr/local/cuda/include \
          -I$OPTICKS_PREFIX/externals/glm/glm \
          -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
    echo build DONE
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

