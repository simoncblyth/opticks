#!/bin/bash 
usage(){ cat << EOU
CSGFoundry_IntersectPrimTest 
=============================

Small geometry testing intersection using CSGMaker 
created CSGSolid/CSGPrim/CSGNode

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
bin=CSGFoundry_IntersectPrimTest

#source $HOME/.opticks/GEOM/GEOM.sh 

geom=JustOrb
#geom=DifferenceBoxSphere
export GEOM=$geom      # see CSGMaker::make for allowable names

export FOLD=/tmp/$USER/opticks/$bin
mkdir -p $FOLD

vars="BASH_SOURCE bin GEOM FOLD"

loglevel(){
   export CSGFoundry=INFO
   #export CSGImport=INFO
}
loglevel


#defarg=info_run_ana
defarg=info_run
arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi 

if [ "${arg/ana}" != "$arg" ]; then 
   ${IPYTHON:-ipython} --pdb -i $script 
   [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 1 
fi 

exit 0

