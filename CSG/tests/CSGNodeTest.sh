#!/bin/bash -l 

bin=CSGNodeTest 
source $HOME/.opticks/GEOM/GEOM.sh 

vars="BASH_SOURCE bin GEOM OPTICKS_T_GEOM ${GEOM}_CFBaseFromGEOM HOME"
for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done

$bin

exit 0 
