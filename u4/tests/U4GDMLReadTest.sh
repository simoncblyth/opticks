#!/bin/bash -l
usage(){ cat << EOU
U4GDMLReadTest.sh
===================

::

   ~/opticks/u4/tests/U4GDMLReadTest.sh


EOU
}

cd $(dirname $BASH_SOURCE)

source $HOME/.opticks/GEOM/GEOM.sh  

vars="BASH_SOURCE GEOM bin"

bin=U4GDMLReadTest 

defarg="info_run"
arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/run}" != "$arg" ]; then
   $bin 
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1  
fi 

if [ "${arg/dbg}" != "$arg" ]; then
   dbg__ $bin 
   [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi 

exit 0 





