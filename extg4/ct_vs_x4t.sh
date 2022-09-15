#!/bin/bash -l 
usage(){ cat << EOU
ct_vs_x4t.sh : compare ct.sh geometry from CSGSimtraceTest with x4t.sh geometry from X4SimtraceTest 
=====================================================================================================

See also:

* extg4/x4t.sh 
* CSG/ct.sh 

EOU
}

arg=${1:-ana}

cbin=CSGSimtraceTest
xbin=X4SimtraceTest

export SYMBOLS="S,T"


geom=nmskSolidMaskVirtual

s_geom=$geom
t_geom=$geom
opt=U1

export S_GEOM=${s_geom}__$opt
export T_GEOM=${t_geom}__$opt
export S_FOLD=/tmp/$USER/opticks/GEOM/$S_GEOM/$cbin/ALL
export T_FOLD=/tmp/$USER/opticks/GEOM/$T_GEOM/$xbin/ALL
export S_LABEL="S:$cbin"
export T_LABEL="T:$xbin"


#export FOCUS=257,-39,7
export TOPLINE="extg4/ct_vs_x4t.sh S_GEOM $S_GEOM T_GEOM $T_GEOM"
export BOTLINE="FOCUS $FOCUS"


if [ "${arg/info}" != "$arg" ]; then 
    vars="BASH_SOURCE arg cbin xbin S_GEOM T_GEOM S_FOLD T_FOLD"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done 
fi

if [ "${arg/ana}"  != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/tests/$xbin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana interactive error && exit 3
fi 

if [ "$arg" == "mpcap" -o "$arg" == "mppub" ]; then

    FOLD=$S_FOLD
    GEOM="${S_GEOM}_${T_GEOM}"

    export CAP_BASE=$FOLD/figs
    export CAP_REL=ct_vs_x4t
    export CAP_STEM=${GEOM}
    case $arg in  
       mpcap) source mpcap.sh cap  ;;  
       mppub) source mpcap.sh env  ;;  
    esac

    if [ "$arg" == "mppub" ]; then 
        source epub.sh 
    fi  
fi 


