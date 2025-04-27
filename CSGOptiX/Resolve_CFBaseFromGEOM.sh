Resolve_CFBaseFromGEOM()
{
   : LOOK FOR CFBase directory containing CSGFoundry geometry
   : HMM COULD PUT INTO GEOM.sh TO AVOID DUPLICATION ? BUT TOO MUCH HIDDEN ?
   : G4CXOpticks_setGeometry_Test GEOM TAKES PRECEDENCE OVER .opticks/GEOM
   : HMM : FOR SOME TESTS WANT TO LOAD GDML BUT FOR OTHERS CSGFoundry
   : to handle that added gdml resolution to eg g4cx/tests/GXTestRunner.sh

   local External_CFBaseFromGEOM=${GEOM}_CFBaseFromGEOMS

   local A_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM
   local B_CFBaseFromGEOM=$TMP/G4CXOpticks_setGeometry_Test/$GEOM
   local C_CFBaseFromGEOM=/cvmfs/opticks.ihep.ac.cn/.opticks/GEOM/$GEOM

   local TestPath=CSGFoundry/prim.npy
   local GDMLPathFromGEOM=$HOME/.opticks/GEOM/$GEOM/origin.gdml

    if [ -d "${!External_CFBaseFromGEOM}" -a -f "${!External_CFBaseFromGEOM}/$TestPath" ]; then
        echo $BASH_SOURCE : USING EXTERNALLY SETUP GEOMETRY ENVIRONMENT : EG FROM OJ DISTRIBUTION
    elif [ -d "$A_CFBaseFromGEOM" -a -f "$A_CFBaseFromGEOM/$TestPath" ]; then
        export ${GEOM}_CFBaseFromGEOM=$A_CFBaseFromGEOM
        #echo $BASH_SOURCE : FOUND A_CFBaseFromGEOM $A_CFBaseFromGEOM containing $TestPath
    elif [ -d "$B_CFBaseFromGEOM" -a -f "$B_CFBaseFromGEOM/$TestPath" ]; then
        export ${GEOM}_CFBaseFromGEOM=$B_CFBaseFromGEOM
        #echo $BASH_SOURCE : FOUND B_CFBaseFromGEOM $B_CFBaseFromGEOM containing $TestPath
    elif [ -d "$C_CFBaseFromGEOM" -a -f "$C_CFBaseFromGEOM/$TestPath" ]; then
        export ${GEOM}_CFBaseFromGEOM=$C_CFBaseFromGEOM
        #echo $BASH_SOURCE : FOUND C_CFBaseFromGEOM $C_CFBaseFromGEOM containing $TestPath
    elif [ -f "$GDMLPathFromGEOM" ]; then
        export ${GEOM}_GDMLPathFromGEOM=$GDMLPathFromGEOM
        echo $BASH_SOURCE : FOUND GDMLPathFromGEOM $GDMLPathFromGEOM
    else
        echo $BASH_SOURCE : NOT-FOUND A_CFBaseFromGEOM $A_CFBaseFromGEOM containing $TestPath
        echo $BASH_SOURCE : NOT-FOUND B_CFBaseFromGEOM $B_CFBaseFromGEOM containing $TestPath
        echo $BASH_SOURCE : NOT-FOUND C_CFBaseFromGEOM $C_CFBaseFromGEOM containing $TestPath
        echo $BASH_SOURCE : NOT-FOUND GDMLPathFromGEOM $GDMLPathFromGEOM
    fi
}
Resolve_CFBaseFromGEOM


