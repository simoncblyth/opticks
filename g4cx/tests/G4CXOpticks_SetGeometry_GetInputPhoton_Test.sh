#!/bin/bash -l 
usage(){ cat << EOU
G4CXOpticks_SetGeometry_GetInputPhoton_Test.sh
===============================================

Test of geometry conversions in isolation. 

EOU
}

#source $(dirname $BASH_SOURCE)/../../bin/GEOM_.sh   # change the geometry with geom_ 
source $HOME/.opticks/GEOM/GEOM.sh 

export GProperty_SIGINT=1
#export NTreeBalance__UnableToBalance_SIGINT=1

logging(){
   export Dummy=INFO
   export G4CXOpticks=INFO
   #export X4PhysicalVolume=INFO
   #export SOpticksResource=INFO
   export CSGFoundry=INFO
   export GSurfaceLib=INFO
}
[ -n "$LOG" ] && logging && env | grep =INFO


export OPTICKS_INPUT_PHOTON=DownXZ1000_f8.npy
#export MOI=Hama:0:1000

bin=G4CXOpticks_SetGeometry_GetInputPhoton_Test

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
    dbg__ $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi 

exit 0



