#!/bin/bash -l 

usage(){ cat << EOU
rundbg.sh
===========

EOU
}


msg="=== $BASH_SOURCE :"

source $OPTICKS_HOME/bin/geocache_hookup.sh

CSG_GGeoTest


#--savegparts --earlyexit $*


