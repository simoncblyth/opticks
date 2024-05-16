#!/bin/bash -l

usage(){ cat << EOU
U4SolidTest.sh
===============

~/o/u4/tests/U4SolidTest.sh 

EOU
}

path=$(realpath $BASH_SOURCE)
bin=U4SolidTest

defarg="run"
arg=${1:-$defarg}


#source $HOME/.opticks/GEOM/GEOM.sh 

#GEOM=BoxGridMultiUnion10_30_YX   ## see U4SolidMaker
#GEOM=LocalFastenerAcrylicConstruction8
#GEOM=OrbOrbMultiUnionSimple
GEOM=OrbOrbMultiUnionSimple2

export U4SolidTest__MAKE=$GEOM
export U4SolidTest__Convert_level=${LEVEL:-4}

test=MAKE
export TEST=${TEST:-$test}

vars="BASH_SOURCE path bin arg GEOM U4SolidTest__MAKE U4SolidTest__Convert_level TEST"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done
fi 


if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run fail && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg fail && exit 2 
fi 

exit 0 


