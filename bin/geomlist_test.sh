#!/bin/bash -l 

usage(){ cat << EOU
geomlist_test.sh 
==================

Source this to check existance of geomlist folders 
and expected files such as simtrace.npy and sframe.npy 


EOU
}


#bin=CSGSimtraceTest
bin=X4SimtraceTest

geomlist_FOLD=/tmp/$USER/opticks/GEOM/%s/$bin/ALL
geomlist_OPT=U1
source $(dirname $BASH_SOURCE)/geomlist.sh 




