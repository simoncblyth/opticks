#!/bin/bash -l 
usage(){ cat << EOU
G4CXOpticks_setGeometry_Test.sh
===================================

Test of geometry conversions in isolation. 

EOU
}

source $(dirname $BASH_SOURCE)/../../bin/GEOM_.sh   # change the geometry with geom_ 

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
#loglevels

env | grep =INFO


#export OPTICKS_INPUT_PHOTON=DownXZ1000_f8.npy
#export MOI=Hama:0:1000

bin=G4CXOpticks_setGeometry_Test

export FOLD=/tmp/$USER/opticks/$bin
mkdir -p $FOLD

defarg=run_ana
arg=${1:-$defarg}

if [ "${arg/run}" != "$arg" ]; then 

    $bin
    MOI=Hama:0:1000 $bin
    MOI=NNVT:0:1000 $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    export TAIL="-o run"
    case $(uname) in 
       Darwin) lldb__ $bin  ;; 
       Linux)  gdb__  $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi 

exit 0



