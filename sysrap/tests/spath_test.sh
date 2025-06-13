#!/bin/bash
usage(){ cat << EOU
spath_test.sh
==============

TEST=Resolve3 ~/opticks/sysrap/tests/spath_test.sh

UNSET=1 TEST=CFBaseFromGEOM ~/opticks/sysrap/tests/spath_test.sh


EOU
} 

cd $(dirname $(realpath $BASH_SOURCE))
source dbg__.sh
export SDIR=$PWD

tmp=/tmp/$USER/opticks
export TMP=${TMP:-$tmp}

name=spath_test 
export FOLD=$TMP/$name

bin=$FOLD/$name
mkdir -p $FOLD

#test=Filesize
#test=ALL
test=last_write_time

export TEST=${TEST:-$test}

export EXECUTABLE=$bin

source $HOME/.opticks/GEOM/GEOM.sh 

# could do the below in GEOM.sh but leave here in repo to be more explicit 
if [ -n "$GEOM" -a -z "${GEOM}_CFBaseFromGEOM" ]; then
    if [ -f "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/prim.npy" ]; then
        export ${GEOM}_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM
    fi 
fi


if [ -n "$UNSET" ]; then
  unset ${GEOM}_CFBaseFromGEOM 
  unset GEOM
fi 


export hello_GEOMSub="yep"



defarg=info_build_run
arg=${1:-$defarg}


if [ "${arg/info}" != "$arg" ]; then 
   vars="BASH_SOURCE SDIR PWD TMP FOLD name bin GEOM TMP TEST"
   for var in $vars ; do printf "%25s : %s\n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
   gcc $name.cc -g -std=c++17 -lstdc++ -lm -I.. -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi

if [ "${arg/dbg}" != "$arg" ]; then 
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3 
fi


exit 0 

