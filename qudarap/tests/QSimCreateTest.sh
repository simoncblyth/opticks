#!/bin/bash 

usage(){ cat << EOU
QSimCreateTest.sh
==================

Note that this fails when REQUIRE_PMT::

   export QSim__REQUIRE_PMT=1

EOU
}

source ~/.opticks/GEOM/GEOM.sh 

defarg="info_run"
arg=${1:-$defarg}
bin=QSimCreateTest

vv="BASH_SOURCE defarg arg bin GEOM ${GEOM}_CFBaseFromGEOM"

if [ "${arg/info}" != "$arg" ]; then
   for v in $vv ; do printf "%20s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then
   source dbg__.sh 
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi 

exit 0 




