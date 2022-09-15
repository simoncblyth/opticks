#!/bin/bash -l 
usage(){ cat << EOU
cf_x4t.sh
===========

See also x4t.sh that loads and plots from a single folder

::

   FOCUS=257,-39,7 ./cf_x4t.sh ana

EOU
}

arg=${1:-ana}
bin=X4SimtraceTest
log=$bin.log

export SYMBOLS="S,T"



#s_geom=nmskSolidMask
#t_geom=nmskSolidMaskTail



s_geom=nmskTailOuter
t_geom=nmskTailInner
opt=U1

export S_GEOM=${s_geom}__$opt
export T_GEOM=${t_geom}__$opt
export S_FOLD=/tmp/$USER/opticks/GEOM/$S_GEOM/$bin/ALL
export T_FOLD=/tmp/$USER/opticks/GEOM/$T_GEOM/$bin/ALL

export FOCUS=257,-39,7

export TOPLINE="extg4/cf_x4t.sh S_GEOM $S_GEOM T_GEOM $T_GEOM"
export BOTLINE="FOCUS $FOCUS"


if [ "${arg/info}" != "$arg" ]; then 
    vars="BASH_SOURCE arg bin S_GEOM T_GEOM S_FOLD T_FOLD"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done 
fi

if [ "${arg/ana}"  != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/tests/$bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana interactive error && exit 3
fi 

if [ "$arg" == "mpcap" -o "$arg" == "mppub" ]; then

    FOLD=$S_FOLD
    GEOM="${S_GEOM}_${T_GEOM}"

    export CAP_BASE=$FOLD/figs
    export CAP_REL=cf_x4t
    export CAP_STEM=${GEOM}
    case $arg in  
       mpcap) source mpcap.sh cap  ;;  
       mppub) source mpcap.sh env  ;;  
    esac

    if [ "$arg" == "mppub" ]; then 
        source epub.sh 
    fi  
fi 


