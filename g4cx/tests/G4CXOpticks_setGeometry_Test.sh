#!/bin/bash -l 
usage(){ cat << EOU
G4CXOpticks_setGeometry_Test.sh
===================================

Test of geometry conversions in isolation. 

EOU
}

export GEOM=ntds3
export ntds3_GDMLPathFromGEOM=/tmp/$USER/opticks/GEOM/$GEOM/G4CXOpticks/origin.gdml

export GProperty_SIGINT=1
#export NTreeBalance__UnableToBalance_SIGINT=1

loglevels(){
   export G4CXOpticks=INFO
   export X4PhysicalVolume=INFO
   export SOpticksResource=INFO
}
loglevels
env | grep =INFO

bin=G4CXOpticks_setGeometry_Test

export TAIL="-o run"

case $(uname) in 
   Darwin) lldb__ $bin  ;; 
   Linux)  gdb__  $bin ;;
esac

