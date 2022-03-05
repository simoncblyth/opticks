#!/bin/bash -l

usage(){ cat << EOU
GPartsTest.sh
==============

Before running this::

   g4ok
   ./G4OKVolumeTest.sh     

       ## gets a G4VPhysicalVolume and translates that with G4Opticks into a geocache
       ## Opticks::writeGeocacheScript writes eg /usr/local/opticks/geocache/geocache.sh
       ## which gets sourced by subsequent "source $OPTICKS_HOME/bin/geocache_hookup.sh"
   
   cg
   ./rundbg.sh
     
       ## sources $OPTICKS_HOME/bin/geocache_hookup.sh and then runs CSG_GGeoTest 
       ## which converts the geocache into CSGFoundry 


    ggeo
    ./GPartsTest.sh  


    cx
    ./cxr_debug.sh 

    EYE=1,0,0 ./cxr_debug.sh 


EOU
}


msg="=== $BASH_SOURCE :"

source $OPTICKS_HOME/bin/geocache_hookup.sh ${1:-new}

${IPYTHON:-ipython} -i --pdb --  GPartsTest.py



