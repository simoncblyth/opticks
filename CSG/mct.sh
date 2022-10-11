#!/bin/bash -l 
usage(){ cat << EOU
mct.sh : intersection and plotting of multiple solids/volumes
===================================================================

See also ct.sh that loads and plots single geometries. 

Workflow:

1. configure GEOM list with *geomlist_* bash function
2. check status of the listed GEOM names::

   ~/opticks/CSG/mct.sh status  

3. if any GEOM are listed as NO-CSGFoundry do translations to create them::

   ~/opticks/CSG/mct.sh translate

4. if any GEOM are listed as NO-Intersect run the intersects::

   ~/opticks/CSG/mct.sh run

4. present the intersects with python plots::

   ~/opticks/CSG/mct.sh ana


Alternatively do all the above with::

    ~/opticks/CSG/mct.sh translate_run_ana 

EOU
}

arg=${1:-ana}
bin=CSGSimtraceTest
log=$bin.log

geomlist_FOLD=/tmp/$USER/opticks/GEOM/%s/$bin/ALL
geomlist_OPT=U1
source $(dirname $BASH_SOURCE)/../bin/geomlist.sh export

#export FOCUS=257,-39,7
#export FOCUS=100,-180,40

export TOPLINE="CSG/mct.sh $geomlist_LABEL $SYMBOLS"
export BOTLINE="FOCUS $FOCUS"


if [ "${arg/translate}" != "$arg" ]; then 
    $(dirname $BASH_SOURCE)/../GeoChain/mtranslate.sh 
fi 

if [ "${arg/info}" != "$arg" ]; then 
    vars="BASH_SOURCE arg bin S_LABEL T_LABEL S_FOLD T_FOLD"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done 
fi

if [ "${arg/status}"  != "$arg" ]; then 
    names=$(source $(dirname $BASH_SOURCE)/../bin/geomlist.sh names)
    for geom in $names 
    do  
         echo $geom 
    done
fi 

if [ "${arg/run}"  != "$arg" ]; then 
    names=$(source $(dirname $BASH_SOURCE)/../bin/geomlist.sh names)
    for geom in $names 
    do  
       export GEOM=$geom 
       export ${GEOM}_CFBaseFromGEOM=/tmp/$USER/opticks/GEOM/$geom    
       # doing what bin/GEOM_.sh normally does
       # could source GEOM_.sh with GEOM defined but its simpler to duplicate this little thing  
       $bin
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
    export CAP_REL=mct
    export CAP_STEM=${LABEL}_${SYMBOLS}
    case $arg in  
       mpcap) source mpcap.sh cap  ;;  
       mppub) source mpcap.sh env  ;;  
    esac

    if [ "$arg" == "mppub" ]; then 
        source epub.sh 
    fi  
fi 


