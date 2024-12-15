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

vars="0 BASE GEOM TAR"

defarg="info_tar"

arg=${1:-$defarg}


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

TAR=$BASE/$GEOM.tar


if [ -f "$TAR" ]; then
   if [ -n "$VERBOSE" ] ; then 
      tar tvf $TAR
   fi 
   du -h $TAR

   md5sum $TAR
fi 


if [ "${arg/info}" != "${arg}" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/help}" != "${arg}" ]; then
   usage
fi 

if [ "${arg/tar}" != "${arg}" ]; then

    if [ -f "$TAR" ]; then
       echo TAR $TAR exists already
    else
       echo Creating GEOM tarball ${GEOM}.tar
       cd $BASE
       tar cvf ${GEOM}.tar ${GEOM}/*
    fi 
fi 


if [ "${arg/scp}" != "${arg}" ]; then
    if [ -f "$TAR" ]; then
        cmd="scp $TAR L:g/"
        echo $cmd 
        eval $cmd 
   else
        echo $BASH_SOURCE : must create TAR $TAR before can scp  
   fi 

fi 



