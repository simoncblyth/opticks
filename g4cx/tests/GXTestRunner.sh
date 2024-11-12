#!/usr/bin/env bash
usage(){ cat << EOU
GXTestRunner.sh
===================

See sysrap/tests/STestRunner.sh for notes

Caution when using this under ctest it is the installed version 
of the runner that is used, so must build+install gxt before 
a change here will take effect.

EOU
}

EXECUTABLE="$1"
shift
ARGS="$@"


geomscript=$HOME/.opticks/GEOM/GEOM.sh
[ -s $geomscript ] && source $geomscript


Resolve_GDMLPathFromGEOM()
{   
   local origin=$HOME/.opticks/GEOM/$GEOM/origin.gdml 
   if [ -f "$origin" ]; then 
        export ${GEOM}_GDMLPathFromGEOM=$origin
        echo $BASH_SOURCE : FOUND origin $origin
   else 
        echo $BASH_SOURCE : NOT-FOUND origin $origin
   fi  
}
Resolve_GDMLPathFromGEOM



vars="HOME PWD GEOM ${GEOM}_GDMLPathFromGEOM BASH_SOURCE EXECUTABLE ARGS"
for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done 

#env 
$EXECUTABLE $@
[ $? -ne 0 ] && echo $BASH_SOURCE : FAIL from $EXECUTABLE && exit 1 

exit 0

