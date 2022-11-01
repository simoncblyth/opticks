#!/bin/bash -l 
usage(){ cat << EOU
G4CXOpticks_setGeometry_Test.sh
===================================

Test of geometry conversions in isolation. 

EOU
}


export GEOM=J004G
export J004G_GDMLPathFromGEOM=$HOME/.opticks/GEOM/J004/origin.gdml

#source $(dirname $BASH_SOURCE)/../../bin/GEOM_.sh   # change the geometry with geom_ 


export GProperty_SIGINT=1
#export NTreeBalance__UnableToBalance_SIGINT=1

loglevels(){
   export Dummy=INFO
   export G4CXOpticks=INFO
   #export X4PhysicalVolume=INFO
   #export SOpticksResource=INFO
   export CSGFoundry=INFO
   export GSurfaceLib=INFO
}
loglevels

env | grep =INFO

bin=G4CXOpticks_setGeometry_Test

defarg=run
arg=${1:-$defarg}

if [ "$arg" == "run" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1 
fi 

if [ "$arg" == "dbg" ]; then 
    export TAIL="-o run"
    case $(uname) in 
       Darwin) lldb__ $bin  ;; 
       Linux)  gdb__  $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi 

exit 0



