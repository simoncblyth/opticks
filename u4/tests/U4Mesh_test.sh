#!/bin/bash -l 
usage(){ cat << EOU
U4Mesh_test.sh
================

Instanciates G4Orb and persists as U4Mesh into /tmp/U4Mesh_test::

    ~/o/u4/tests/U4Mesh_test.sh
    SOLID=Tubs ~/o/u4/tests/U4Mesh_test.sh ana


    ~/o/u4/tests/U4Mesh_test.sh view


EOU
}

cd $(dirname $BASH_SOURCE)
name=U4Mesh_test
export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

clhep-
g4-

#solid=Torus
#solid=Orb
#solid=Box
#solid=Tet
solid=Tubs
#solid=Cons


allsolid="Orb Box Tet Tubs Cons Torus"

export SOLID=${SOLID:-$solid}


vars="BASH_SOURCE name FOLD bin SOLID"

defarg="info_build_run_ana"
arg=${1:-$defarg}


[ "${arg/info}" != "$arg" ] && for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 


if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc \
         -I.. \
         -std=c++11 -lstdc++ \
         -I$HOME/opticks/sysrap \
         -I$(clhep-prefix)/include \
         -I$(g4-prefix)/include/Geant4  \
         -L$(g4-libdir) \
         -L$(clhep-prefix)/lib \
         -lG4global \
         -lG4geometry \
         -lG4graphics_reps \
         -lCLHEP \
         -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi 

if [ "${arg/all}" != "$arg" ]; then
    for solid in $allsolid ; do echo $BASH_SOURCE : $solid && SOLID=$solid $bin ; done  
    [ $? -ne 0 ] && echo $BASH_SOURCE all error && exit 2
fi 

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
fi 

if [ "${arg/view}" != "$arg" ]; then

    export FOLD=/data/blyth/opticks/U4TreeCreateTest/stree/mesh
    export SOLID=HamamatsuR12860sMask_virtual0xa0b8450

    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
fi 






exit 0 

