#!/bin/bash
usage(){ cat << EOU
sn_test.sh
==========

~/o/sysrap/tests/sn_test.sh


EOU
}


cd $(dirname $(realpath $BASH_SOURCE))
name=sn_test

export FOLD=/tmp/$name
mkdir -p $FOLD

bin=$FOLD/$name
script=$name.py

#defarg="info_build_run_ana"
defarg="info_build_run"
arg=${1:-$defarg}

opt=-DWITH_CHILD

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

#test=ALL
#test=idx_0
#test=create_0
#test=deepcopy_0
#test=deepcopy_2
#test=disown_child_0
#test=set_right_0
#test=list_tree_0
#test=difference_and_list_tree_0
#test=CreateSmallerTreeWithListNode_0
#test=CreateSmallerTreeWithListNode_2

#test=Serialize
test=Serialize_Import

export TEST=${TEST:-$test}

logging()
{
    type $FUNCNAME
    export sn__level=2
    #export s_pool_level=2
    #export sn__GetLVRoot_DUMP=1
}
[ -n "$LOG" ] && logging


vars="BASH_SOURCE bin script opt TEST sn__level"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc \
        ../sn.cc \
        ../s_tv.cc \
        ../s_pa.cc \
        ../s_bb.cc \
        ../s_csg.cc \
        -I.. \
        -I$HOME/np \
        -I$OPTICKS_PREFIX/externals/glm/glm \
        -I$CUDA_PREFIX/include \
        -lm \
        $opt -g -std=c++17 -lstdc++ -o $bin
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

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE pdb error && exit 4
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 5
fi

exit 0

