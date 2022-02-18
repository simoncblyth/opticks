#!/bin/bash -l 

msg="=== $BASH_SOURCE "

catgeom=$(cat ~/.opticks/GEOM.txt 2>/dev/null | grep -v \#) && [ -n "$catgeom" ] && echo $msg catgeom $catgeom override of default geom $geom && geom=${catgeom%%_*} 
GEOM=${GEOM:-$geom}

echo $msg catgeom $catgeom geom $geom GEOM $GEOM


#ori=0,-70.71067811865476,0

ori=0,0,0
dir=1,0,0

ORI=${ORI:-$ori}
DIR=${DIR:-$dir}

export GEOM 
export ORI  
export DIR  
export DUMP=3


CSGQueryTest O


