#!/bin/bash -l 
usage(){ cat << EOU
cf_x4t.sh
===========

::

   FOCUS=257,-39,7 ./cf_x4t.sh ana

EOU
}

arg=${1:-ana}
bin=X4SimtraceTest
log=$bin.log

export SYMBOLS="S,T"

export S_GEOM=nmskSolidMask
export T_GEOM=nmskSolidMaskTail
export S_FOLD=/tmp/$USER/opticks/$S_GEOM/$bin/ALL
export T_FOLD=/tmp/$USER/opticks/$T_GEOM/$bin/ALL

export FOCUS=257,-39,7



if [ "${arg/info}" != "$arg" ]; then 
    vars="BASH_SOURCE arg bin S_GEOM T_GEOM S_FOLD T_FOLD"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done 
fi

if [ "${arg/ana}"  != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/tests/$bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana interactive error && exit 3
fi 


