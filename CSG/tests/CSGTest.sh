#!/bin/bash -l 
usage(){ cat << EOU
CSGTest.sh
===========

::

   BIN=CSGNodeTest     ~/o/CSG/tests/CSGTest.sh
   BIN=CSGPrimSpecTest ~/o/CSG/tests/CSGTest.sh
   BIN=CSGPrimTest     ~/o/CSG/tests/CSGTest.sh
   BIN=CSGFoundryTest  ~/o/CSG/tests/CSGTest.sh

   BIN=CSGNodeTest BP=sframe::~sframe  ~/o/CSG/tests/CSGTest.sh


EOU
}

bin=CSGNodeTest
BIN=${BIN:-$bin}
source $HOME/.opticks/GEOM/GEOM.sh 

vars="BASH_SOURCE BIN GEOM OPTICKS_T_GEOM ${GEOM}_CFBaseFromGEOM HOME"
for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done

dbg__ $BIN

exit 0 
