#!/bin/bash -l 
usage(){ cat << EOU
CSGFoundry_getMeshName_Test.sh
=============================

Loads a CSGFoundry geometry specified by GEOM envvar and 

EOU
}


DIR=$(dirname $BASH_SOURCE)

source ~/.opticks/GEOM/GEOM.sh  # sets GEOM envvar, edit with GEOM bash function

export ${GEOM}_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM


msg="=== $BASH_SOURCE :"
bin=CSGFoundry_getMeshName_Test
defarg="run_ana"
arg=${1:-$defarg}


if [ "${arg/run}" != "$arg" ]; then 
   $bin
   [ $? -ne 0 ] && echo $msg run $bin error && exit 1
fi 

exit 0 

