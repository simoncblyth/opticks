#!/bin/bash -l

usage(){ cat << EOU
U4SolidTest.sh
===============

~/o/u4/tests/U4SolidTest.sh 

EOU
}

bin=U4SolidTest

defarg="run"
arg=${1:-$defarg}

geom=BoxGridMultiUnion10:30_YX   ## see U4SolidMaker
#geom=LocalFastenerAcrylicConstruction8

export U4SolidTest__MAKE=$geom  
export U4SolidTest__Convert_level=${LEVEL:-3}

test=MAKE
export TEST=${TEST:-$test}



if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run fail && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg fail && exit 2 
fi 

exit 0 


