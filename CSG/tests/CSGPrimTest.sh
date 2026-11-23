#!/bin/bash
usage(){ cat << EOU

~/o/CSG/tests/CSGPrimTest.sh

EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

source $HOME/.opticks/GEOM/GEOM.sh
#export ${GEOM}_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM

name=CSGPrimTest
bin=$name


$bin
[ $? -ne 0 ] && echo run error && exit 2

exit 0

