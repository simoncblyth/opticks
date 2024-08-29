#!/bin/bash 
usage(){  cat << EOU
GEOM_tar.sh
==============

This is primarily for expert use, so it does not need to 
be included in the install. 

~/opticks/bin/GEOM_tar.sh 
   creates tarball for current GEOM directory if it does not already exist 

~/opticks/bin/GEOM_tar.sh info 
   dumps vars
 
~/opticks/bin/GEOM_tar.sh help 
   dumps this usage message 
 
EOU
}


BASE=$HOME/.opticks/GEOM

source $BASE/GEOM.sh 

vars="0 BASE GEOM"

defarg="info_run"

arg=${1:-$defarg}

if [ "${arg/info}" != "${arg}" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/help}" != "${arg}" ]; then
   usage
fi 

if [ "${arg/run}" != "${arg}" ]; then

    if [ -z "$GEOM" ]; then 
       echo ERROR - GEOM envvar is not defined
       exit 1 
    else
       echo GEOM $GEOM is defined
    fi 

    if [ ! -d "$BASE/$GEOM" ]; then
       echo ERROR - GEOM $GEOM directory $BASE/$GEOM does not exist  
    else
       echo GEOM $GEOM directory $BASE/$GEOM exists
    fi 

    if [ -f "$BASE/$GEOM.tar" ]; then
       echo GEOM $BASE/${GEOM}.tar exists already   
       cd $BASE

       if [ -n "$VERBOSE" ] ; then 
           tar tvf ${GEOM}.tar
       fi 

       du -h ${GEOM}.tar
    else
       echo Creating GEOM tarball ${GEOM}.tar
       cd $BASE
       tar cvf ${GEOM}.tar ${GEOM}/*
    fi 

fi 


