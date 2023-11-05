#!/bin/bash -l 
usage(){ cat << EOU
COMMON.sh 
============

WHY ARE MOVING THIS KIND OF CONFIG TO $HOME/.opticks/GEOM/GEOM.sh 
------------------------------------------------------------------

The content of the below scripts that bin/COMMON.sh sources
are very specific to each users geometry::

   bin/GEOM_.sh 
   bin/OPTICKS_INPUT_PHOTON.sh

Hence it is more appropriate to config both geometry 
and input photons from a user script in a standard location::

   $HOME/.opticks/GEOM/GEOM.sh 

This standard location is used by the opticks-t ctests, 
so user geometry can be checked by the tests. 

Clearly when working with multiple geometries it might
prove move convenient for GEOM.sh to source different
scripts depending on the GEOM envvar value.


Functions
------------

Shortcut bash functions:

com_
    Edit the COMMON.sh config scripts 
geom_
    Edit the GEOM_.sh config
oip
    Edit the OPTICKS_INPUT_PHOTON.sh config 

EOU
}


bindir=$(dirname $BASH_SOURCE)

source $bindir/GEOM_.sh                   # defines and exports : GEOM, GEOMDIR 
source $bindir/OPTICKS_INPUT_PHOTON.sh    # defines and exports : OPTICKS_INPUT_PHOTON

# upfind_cfbase relocated to GEOM_.sh 

case $GEOM in 
 J000) OPTICKS_INPUT_PHOTON_FRAME=NNVT:0:1000 ;;
 J001) OPTICKS_INPUT_PHOTON_FRAME=Hama:0:1000 ;;
 J002) OPTICKS_INPUT_PHOTON_FRAME=NNVT:0:1000 ;;
 J003) OPTICKS_INPUT_PHOTON_FRAME=NNVT:0:1000 ;;
esac

[ -n "$OPTICKS_INPUT_PHOTON_FRAME" ] && export OPTICKS_INPUT_PHOTON_FRAME
[ -n "$OPTICKS_INPUT_PHOTON_FRAME" ] && export MOI=$OPTICKS_INPUT_PHOTON_FRAME

if [ -z "$QUIET" ]; then 
    vars="BASH_SOURCE GEOM OPTICKS_INPUT_PHOTON OPTICKS_INPUT_PHOTON_FRAME MOI"
    for var in $vars ; do printf "%30s : %s\n" $var ${!var} ; done 
fi 

