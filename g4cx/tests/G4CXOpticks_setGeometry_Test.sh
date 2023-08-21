#!/bin/bash -l 
usage(){ cat << EOU
G4CXOpticks_setGeometry_Test.sh
===================================

Test of geometry conversions in isolation::
                  
    OLD : Geant4 --[X4]--> GGeo ----[CSG_GGeo]--->  CSGFoundry 
    NEW : Geant4 --[U4]--> SSim/stree --[CSGImport]-> CSGFoundry 

CAUTIONS:

1. runs from GDML, so SensitiveDetector info is lost

   * SO NOT USEFUL FOR CHECKING SENSORS

2. this currently does the old workflow and some parts of the new workflow

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)  

defarg=info_dbg_ana
arg=${1:-$defarg}

bin=G4CXOpticks_setGeometry_Test
script=$SDIR/$bin.py 

export FOLD=/tmp/$USER/opticks/$bin
mkdir -p $FOLD

source $HOME/.opticks/GEOM/GEOM.sh   # sets GEOM envvar 

case $GEOM in 
   FewPMT) geomscript=$SDIR/../../u4/tests/FewPMT.sh ;;
esac

origin=$HOME/.opticks/GEOM/$GEOM/origin.gdml

vars="BASH_SOURCE arg SDIR GEOM FOLD bin geomscript script origin"

if [ -f "$origin" ]; then
    export ${GEOM}_GDMLPathFromGEOM=$origin
fi 

if [ -n "$geomscript" -a -f "$geomscript" ]; then 
    echo $BASH_SOURCE : GEOM $GEOM : sourcing geomscript $geomscript
    source $geomscript
else
    echo $BASH_SOURCE : GEOM $GEOM : no geomscript    
fi 

export GProperty_SIGINT=1
#export NTreeBalance__UnableToBalance_SIGINT=1
#export BFile__preparePath_SIGINT=1
#export GGeo__save_SIGINT=1

#savedir=~/.opticks/GEOM/$GEOM
#savedir=/tmp/GEOM/$GEOM
#export SAVEDIR=${SAVEDIR:-$savedir}

export G4CXOpticks__setGeometry_saveGeometry=$FOLD
export G4CXOpticks__saveGeometry_saveGGeo=1

#export NNodeNudger__DISABLE=1
#export X4Solid__convertPolycone_nudge_mode=0 # 0:DISABLE 

#export U4Polycone__DISABLE_NUDGE=1 


#export s_csg_level=2 
#export sn__level=2




#export U4Tree__DISABLE_OSUR_IMPLICIT=1

loglevels(){
   export Dummy=INFO
   export G4CXOpticks=INFO
   #export X4PhysicalVolume=INFO
   #export SOpticksResource=INFO
   export CSGFoundry=INFO
   export GSurfaceLib=INFO
   export U4VolumeMaker=INFO
   #export NCSG=INFO
}
#loglevels
env | grep =INFO


if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
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
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi 

exit 0

