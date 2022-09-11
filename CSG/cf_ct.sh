#!/bin/bash -l 
usage(){ cat << EOU
cf_ct.sh
===========

See also ct.sh that loads and plots from a single folder

EOU
}

arg=${1:-ana}
bin=CSGSimtraceTest
log=$bin.log

export SYMBOLS="S,T"

#s_geom=nmskSolidMask
#t_geom=nmskSolidMaskTail

s_geom=nmskTailOuter
t_geom=nmskTailInner
opt=U1

export S_GEOM=${s_geom}__$opt
export T_GEOM=${t_geom}__$opt
export S_FOLD=/tmp/$USER/opticks/$S_GEOM/$bin/ALL
export T_FOLD=/tmp/$USER/opticks/$T_GEOM/$bin/ALL

export FOCUS=257,-39,7

export TOPLINE="CSG/cf_ct.sh S_GEOM $S_GEOM T_GEOM $T_GEOM"
export BOTLINE="FOCUS $FOCUS"


if [ "${arg/info}" != "$arg" ]; then 
    vars="BASH_SOURCE arg bin S_GEOM T_GEOM S_FOLD T_FOLD"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done 
fi

if [ "${arg/ana}"  != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/tests/$bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana interactive error && exit 3
fi 


