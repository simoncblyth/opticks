#!/bin/bash -l 

bin=CSGNodeTest 
source $HOME/.opticks/GEOM/GEOM.sh 
export ${GEOM}_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM


vars="BASH_SOURCE bin GEOM OPTICKS_T_GEOM"
for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done

$bin


exit 0 

