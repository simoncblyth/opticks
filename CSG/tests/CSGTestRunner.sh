#!/usr/bin/env bash
usage(){ cat << EOU
CSGTestRunner.sh
===================

See sysrap/tests/STestRunner.sh for notes

EOU
}

EXECUTABLE="$1"
shift
ARGS="$@"



geomscript=$HOME/.opticks/GEOM/GEOM.sh
[ -s $geomscript ] && source $geomscript


vars="HOME PWD GEOM BASH_SOURCE EXECUTABLE ARGS"
for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done 

#env 
$EXECUTABLE $@
[ $? -ne 0 ] && echo $BASH_SOURCE : FAIL from $EXECUTABLE && exit 1 

exit 0

