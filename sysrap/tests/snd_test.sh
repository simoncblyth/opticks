#!/bin/bash -l 

name=snd_test 

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

#test=save
#test=load
#test=max_depth
#test=num_node
#test=inorder
#test=dump
test=render

#tree=0
#tree=1
#tree=2
#tree=3
tree=4

export TEST=${TEST:-$test} 
export TREE=${TREE:-$tree}

defarg="build_run"
case $TEST in 
    save|load) defarg="build_run_ana"
esac

arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc ../snd.cc ../scsg.cc \
        -g -std=c++11 -lstdc++ -Wsign-compare -Wunused-variable \
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

exit 0 


