#!/bin/bash -l 

usage(){ cat << EOU
~/o/u4/tests/U4MaterialTest.sh 
===============================

EOU
}

source $HOME/.opticks/GEOM/GEOM.sh 
bin=U4MaterialTest 

defarg=run
arg=${1:-$defarg}

if [ "${arg/run}" != "$arg" ]; then  
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then  
   gdb__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2 
fi 

exit 0 


