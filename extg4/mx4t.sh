#!/bin/bash -l 
usage(){ cat << EOU
mx4t.sh
===========

See also x4t.sh that loads and plots from a single folder

and CSG/mct.sh for CSG equivalent. 

EOU
}

arg=${1:-ana}
bin=X4SimtraceTest
log=$bin.log

geomlist_FOLD=/tmp/$USER/opticks/GEOM/%s/$bin/ALL
geomlist_OPT=U1
source $(dirname $BASH_SOURCE)/../bin/geomlist.sh        

#export FOCUS=257,-39,7

export TOPLINE="CSG/mx4t.sh $geomlist_LABEL $SYMBOLS"
export BOTLINE="FOCUS $FOCUS"

if [ "${arg/info}" != "$arg" ]; then 
    vars="BASH_SOURCE arg bin S_LABEL T_LABEL S_FOLD T_FOLD"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done 
fi

if [ "${arg/ana}"  != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/tests/$bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana interactive error && exit 3
fi

if [ "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    FOLD=$S_FOLD
    if [ -n "$geomlist_LABEL" ]; then 
        LABEL=${geomlist_LABEL}
    else
        LABEL=${S_LABEL}
    fi 

    export CAP_BASE=$FOLD/figs
    export CAP_REL=mx4t
    export CAP_STEM=${LABEL}_${SYMBOLS}
    case $arg in
       mpcap) source mpcap.sh cap  ;;
       mppub) source mpcap.sh env  ;;
    esac

    if [ "$arg" == "mppub" ]; then
        source epub.sh
    fi
fi


