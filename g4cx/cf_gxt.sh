#!/bin/bash -l 

defarg="ana"
arg=${1:-$defarg}


export T_GEOM=nmskTailOuter
export S_GEOM=nmskTailInner

export T_FOLD=/tmp/$USER/opticks/GeoChain/$T_GEOM/G4CXSimtraceTest/ALL
export S_FOLD=/tmp/$USER/opticks/GeoChain/$S_GEOM/G4CXSimtraceTest/ALL

if [ "info" == "$arg" ]; then
    vars="BASH_SOURCE arg defarg T_GEOM S_GEOM T_FOLD S_FOLD"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done 
fi 

if [ "ana" == "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/tests/cf_G4CXSimtraceTest.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 1 
fi 


exit 0 

