#!/bin/bash 
usage(){ cat << EOU
G4CX_U4TreeCreateCSGFoundryTest.sh
===================================

Creates Geant4 PV configured with GEOM envvar, 
converts to Opticks stree/CSGFoundry and persists
the CSGFoundry into ~/.opticks/GEOM.
 
Visualize the result with ~/o/cxr_min.sh 

EOU
}

name=G4CX_U4TreeCreateCSGFoundryTest

source $HOME/.opticks/GEOM/GEOM.sh 
fold=$HOME/.opticks/GEOM/$GEOM
export FOLD=$fold

if [ -d "$FOLD" ]; then
   echo $BASH_SOURCE FOLD $FOLD exists already - skipping  
else
   $name 
   [ $? -ne 0 ] && echo $BASH_SOURCE - run error && exit 1
fi  

exit 0 


