#!/bin/bash -l 

GEOM=${GEOM:-ntds3}
BASE=/tmp/$USER/opticks/GEOM/$GEOM/G4CXOpticks
export OpticksGDMLPath=$BASE/origin.gdml
export GProperty_SIGINT=1

loglevels(){
   export G4CXOpticks=INFO
   export X4PhysicalVolume=INFO
}
loglevels


lldb__ G4CXOpticks_setGeometry_Test

