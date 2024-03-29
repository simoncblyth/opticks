#!/bin/bash -l 
usage(){ cat << EOU
SMesh_test.sh
==============

::

    ~/o/sysrap/tests/SMesh_test.sh

    SOLID=Tet ~/o/sysrap/tests/SMesh_test.sh run

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

cd $(dirname $(realpath $BASH_SOURCE))

vars="BASH_SOURCE PWD MESH_FOLD"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    glm-
    gcc $name.cc -std=c++11 -lstdc++ -I.. -I$(glm-prefix) -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi 

exit 0 

