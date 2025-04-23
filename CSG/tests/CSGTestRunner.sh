#!/usr/bin/env bash
usage(){ cat << EOU
CSGTestRunner.sh
===================

See sysrap/tests/STestRunner.sh for notes

NB when updating this need to build+install CSG before changes take effect
   on ctest results as the installed runner is used by opticks-t/ctest

EOU
}

EXECUTABLE="$1"
shift
ARGS="$@"



geomscript=$HOME/.opticks/GEOM/GEOM.sh
[ -s $geomscript ] && source $geomscript


Resolve_CFBaseFromGEOM()
{
   : LOOK FOR CFBase directory containing CSGFoundry geometry 
   : HMM COULD PUT INTO GEOM.sh TO AVOID DUPLICATION ?
   : G4CXOpticks_setGeometry_Test GEOM TAKES PRECEDENCE OVER .opticks/GEOM

   local A_CFBaseFromGEOM=$TMP/G4CXOpticks_setGeometry_Test/$GEOM
   local B_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM
   local TestPath=CSGFoundry/prim.npy

    if [ -d "$A_CFBaseFromGEOM" -a -f "$A_CFBaseFromGEOM/$TestPath" ]; then
        export ${GEOM}_CFBaseFromGEOM=$A_CFBaseFromGEOM
        echo $BASH_SOURCE : FOUND A_CFBaseFromGEOM $A_CFBaseFromGEOM containing $TestPath
    elif [ -d "$B_CFBaseFromGEOM" -a -f "$B_CFBaseFromGEOM/$TestPath" ]; then
        export ${GEOM}_CFBaseFromGEOM=$B_CFBaseFromGEOM
        echo $BASH_SOURCE : FOUND B_CFBaseFromGEOM $B_CFBaseFromGEOM containing $TestPath
    else
        echo $BASH_SOURCE : NOT-FOUND A_CFBaseFromGEOM $A_CFBaseFromGEOM containing $TestPath
        echo $BASH_SOURCE : NOT-FOUND B_CFBaseFromGEOM $B_CFBaseFromGEOM containing $TestPath
    fi  
}

if [ -n "$GEOM" -a -n "${GEOM}_CFBaseFromGEOM" ]; then
   echo $BASH_SOURCE - using external config for GEOM $GEOM ${GEOM}_CFBaseFromGEOM 
else
   Resolve_CFBaseFromGEOM
fi



vars="HOME PWD GEOM BASH_SOURCE EXECUTABLE ARGS"
for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done 

#env 
$EXECUTABLE $@
[ $? -ne 0 ] && echo $BASH_SOURCE : FAIL from $EXECUTABLE && exit 1 

exit 0

