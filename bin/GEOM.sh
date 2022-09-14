#!/bin/bash -l 
usage(){ cat << EOU
GEOM.sh : defines and exports GEOM envvar using string obtained from GEOM.txt file
=====================================================================================

To change the string, and hence the GEOM envvar, 
use the "geom" bash function from opticks/opticks.bash

* HMM: notice that "geom" is distinct from "geom_" which does similar 

This script does not make use of opticks bash functions 
as in some situations it is more convenient to simply source a script 
and not rely on the hookup of opticks.bash functions. 

Usage::

    source ../bin/GEOM.sh 

EOU
}


local-opticks-home(){ echo $(dirname $(dirname $BASH_SOURCE)) ; }

local-opticks-geompath()
{
    : uses the first of dotpath or homepath 
    local dotpath=$HOME/.opticks/GEOM.txt
    local homepath=$(local-opticks-home)/GEOM.txt
    local path=""
    if [ -f "$dotpath" ]; then 
        path=$dotpath
    elif [ -f "$homepath" ]; then 
        path=$homepath
    fi   
    echo $path
}

local-opticks-geom () 
{ 
    : cat the geompath excluding comments and when trim arg is used remove portion of string beginning with underscore
    local arg=${1:-trim}
    local geompath=$(local-opticks-geompath)
    local geom=""
    if [ -f "$geompath" ]; then
        local catgeom=$(cat $geompath 2>/dev/null | grep -v \#) 
        if [ -n "$catgeom" ]; then 
            case $arg in 
              trim) geom=$(echo ${catgeom%%_*})  ;;
              asis) geom=$catgeom ;;
                 *) geom=$catgeom ;;
            esac 
        fi 
    fi;
    echo $geom
}


unset GEOM
arg=${1:-asis}
geom=$(local-opticks-geom $arg)
export GEOM=$geom
echo === $BASH_SOURCE : GEOM $GEOM arg $arg local-opticks-home $(local-opticks-home)


