#!/bin/bash 
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


solid=Torus
#solid=Orb
#solid=Box
#solid=Tet
#solid=Tubs
#solid=Cons


export U4Mesh__NumberOfRotationSteps_entityType_G4Torus=48
export TITLE="U4Mesh_test.sh U4Mesh__NumberOfRotationSteps_entityType_G4Torus $U4Mesh__NumberOfRotationSteps_entityType_G4Torus "


# moving away from these 
#clhep-
#g4-

get-cmake-prefix(){ echo $CMAKE_PREFIX_PATH | tr ":" "\n" | grep $1 ; }

if [ "$(uname)" == "Darwin" ]; then
    g4-prefix(){    get-cmake-prefix g4 ; }
    g4-libdir(){    echo $(g4-prefix)/lib ; }
    clhep-prefix(){ get-cmake-prefix clhep ; }
else
    g4-prefix(){    get-cmake-prefix Geant4 ; }
    g4-libdir(){    echo $(g4-prefix)/lib64 ; }
    clhep-prefix(){ get-cmake-prefix CLHEP ; }
fi 

CLHEP_PREFIX=$(clhep-prefix)
G4_PREFIX=$(g4-prefix)
G4_LIBDIR=$(g4-libdir)

if [ "$(uname)" == "Darwin" ]; then
    export DYLD_LIBRARY_PATH=$CLHEP_PREFIX/lib:$G4_LIBDIR 
fi 


allsolid="Orb Box Tet Tubs Cons Torus"

export SOLID=${SOLID:-$solid}


vars="BASH_SOURCE name FOLD bin SOLID CLHEP_PREFIX G4_PREFIX"

defarg="info_build_run_ana"
arg=${1:-$defarg}


[ "${arg/info}" != "$arg" ] && for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 


if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc \
         -I.. \
         -std=c++11 -lstdc++ \
         -I$HOME/opticks/sysrap \
         -I$CLHEP_PREFIX/include \
         -I$G4_PREFIX/include/Geant4  \
         -L$G4_LIBDIR \
         -L$CLHEP_PREFIX/lib \
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

