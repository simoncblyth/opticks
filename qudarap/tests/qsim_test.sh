#!/bin/bash -l 
usage(){ cat << EOU
qsim_test.sh
==============


EOU
}

name=qsim_test 

defarg="build_run"
arg=${1:-$defarg}

export FOLD=/tmp/$name
mkdir -p $FOLD


if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -std=c++11 -lstdc++ \
       -DMOCK_CURAND \
       -I.. \
       -I$OPTICKS_PREFIX/include/SysRap  \
       -I/usr/local/cuda/include \
       -I$OPTICKS_PREFIX/externals/glm/glm \
       -I$OPTICKS_PREFIX/externals/plog/include \
       -o $FOLD/$name 

    [ $? -ne 0 ] && echo $msg build error && exit 1 
fi 


if [ "${arg/run}" != "$arg" ]; then 
    $FOLD/$name
    [ $? -ne 0 ] && echo $msg run error && exit 2 
fi

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $name.py
    [ $? -ne 0 ] && echo $msg ana error && exit 3 
fi


exit 0 



