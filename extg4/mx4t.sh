#!/bin/bash -l 
usage(){ cat << EOU
mx4t.sh : X4SimtraceTest : Geant4 intersection and presentation
==================================================================

Workflow:

1. configure GEOM list with *geomlist_* bash function
2. check status of the listed GEOM names:: 

    ~/opticks/extg4/mx4t.sh status 
    x4 ; ./mx4t.sh status 

3. if any GEOM are listed as NO-Intersect run the intersects::

    ~/opticks/extg4/mx4t.sh run
    x4 ; ./mx4t.sh run

4. present the intersects with python plotting::

    ~/opticks/extg4/mx4t.sh ana
    x4 ; ./mx4t.sh ana

See also:

* x4t.sh that loads and plots from a single folder
* CSG/mct.sh for CSG equivalent. 

EOU
}

arg=${1:-ana}
bin=X4SimtraceTest
log=$bin.log

geomlist_FOLD=/tmp/$USER/opticks/GEOM/%s/$bin/ALL
geomlist_OPT=U1
source $(dirname $BASH_SOURCE)/../bin/geomlist.sh export       

#export FOCUS=257,-39,7

export TOPLINE="extg4/mx4t.sh $geomlist_LABEL $SYMBOLS"
export BOTLINE="FOCUS $FOCUS"

if [ "${arg/status}" != "$arg" ]; then 
    echo geomlist export gives status
fi 

if [ "${arg/info}" != "$arg" ]; then 
    vars="BASH_SOURCE arg bin S_LABEL T_LABEL S_FOLD T_FOLD"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done 
fi


if [ "${arg/run}"  != "$arg" ]; then 
   names=$(source $(dirname $BASH_SOURCE)/../bin/geomlist.sh names)
   for geom in $names ; do 
      echo geom $geom
      GEOM=$geom $bin
   done 
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


