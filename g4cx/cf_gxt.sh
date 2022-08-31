#!/bin/bash -l 
usage(){ cat <<EOU
cf_gxt.sh : compare simtrace from three geometries 
=======================================================

::

   U_OFFSET=0,0,200 ./cf_gxt.sh 

EOU
}

defarg="ana"
arg=${1:-$defarg}

export S_GEOM=nmskSolidMask
export T_GEOM=nmskSolidMaskTail

#export S_GEOM=nmskTailInner
#export T_GEOM=nmskTailOuter
#export U_GEOM=nmskSolidMaskTail

export S_FOLD=/tmp/$USER/opticks/GeoChain/$S_GEOM/G4CXSimtraceTest/ALL
export T_FOLD=/tmp/$USER/opticks/GeoChain/$T_GEOM/G4CXSimtraceTest/ALL
export U_FOLD=/tmp/$USER/opticks/GeoChain/$U_GEOM/G4CXSimtraceTest/ALL

if [ "info" == "$arg" ]; then
    vars="BASH_SOURCE arg defarg S_GEOM T_GEOM U_GEOM S_FOLD T_FOLD U_FOLD"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done 
fi 

if [ "ana" == "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/tests/cf_G4CXSimtraceTest.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 1 
fi 

exit 0 

