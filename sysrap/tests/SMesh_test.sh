#!/bin/bash -l 
usage(){ cat << EOU
SMesh_test.sh
==============

::

    ~/o/sysrap/tests/SMesh_test.sh

    ~/o/sysrap/tests/SScene_test.sh


EOU
}

name=SMesh_test
defarg="info_build_run"
arg=${1:-$defarg}

mkdir -p /tmp/$name
bin=/tmp/$name/$name

#solid=Torus
#solid=Orb
solid=Box
SOLID=${SOLID:-$solid}

mesh_fold=/tmp/U4Mesh_test/$SOLID
export MESH_FOLD=${MESH_FOLD:-$mesh_fold}

scene_fold=/tmp/SScene_test
export SCENE_FOLD=${SCENE_FOLD:-$scene_fold}

vars="BASH_SOURCE PWD MESH_FOLD SCENE_FOLD"

cd $(dirname $(realpath $BASH_SOURCE))


if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    glm-
    gcc $name.cc -g -std=c++11 -lstdc++ -I.. -I$(glm-prefix) -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 3
fi 



exit 0 

