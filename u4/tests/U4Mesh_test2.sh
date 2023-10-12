#!/bin/bash -l 
usage(){ cat << EOU
U4Mesh_test2.sh
================

Expands on U4Mesh_test.sh with addition of dependency
on j/PMTSim to provide access to complex solids::

    ~/opticks/u4/tests/U4Mesh_test2.sh

    GEOM=xjfcSolid ~/opticks/u4/tests/U4Mesh_test2.sh
    GEOM=xjacSolid ~/opticks/u4/tests/U4Mesh_test2.sh

    GEOM=sjclSolid ~/opticks/u4/tests/U4Mesh_test2.sh
    GEOM=sjfxSolid ~/opticks/u4/tests/U4Mesh_test2.sh
    GEOM=sjrcSolid ~/opticks/u4/tests/U4Mesh_test2.sh
    GEOM=sjrfSolid ~/opticks/u4/tests/U4Mesh_test2.sh

    GEOM=facrSolid ~/opticks/u4/tests/U4Mesh_test2.sh


Compare two geometries::

      AGEOM=sjclSub BGEOM=sjclDown  ~/opticks/u4/tests/U4Mesh_test2.sh ana


EOU
}

cd $(dirname $BASH_SOURCE)
name=U4Mesh_test2
BASE=/tmp/$name
bin=$BASE/$name

if [ -n "$AGEOM" -a -n "$BGEOM" ]; then
   script=${name}_cf.py  
   AFOLD=$BASE/$AGEOM
   BFOLD=$BASE/$BGEOM
   export AFOLD
   export BFOLD
elif [ -n "$GEOM" ]; then 
   script=$name.py 
   FOLD=$BASE/$GEOM
   mkdir -p $FOLD
   export FOLD
else
   echo $BASH_SOURCE : ERROR GEOM is not defined 
   exit 1 
fi 

clhep-
g4-

vars="BASH_SOURCE name BASE FOLD bin GEOM AGEOM BGEOM AFOLD BFOLD script"


defarg="info_build_run_ana"
arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then
    gcc \
         $name.cc \
         -I.. \
         -g -std=c++11 -lstdc++ \
         -I$HOME/opticks/sysrap \
         -I$(clhep-prefix)/include \
         -I$(g4-prefix)/include/Geant4  \
         -L$(g4-prefix)/lib \
         -L$(clhep-prefix)/lib \
         -lG4global \
         -lG4geometry \
         -lG4graphics_reps \
         -lCLHEP \
         -I$HOME/j/PMTSim \
         -L$OPTICKS_PREFIX/lib \
         -lPMTSim \
         -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    case $(uname) in
       Darwin) lldb__ $bin ;;
       Linux)  gdb__  $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi 

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi 


if [ "$arg" == "pvcap" -o "$arg" == "pvpub" -o "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    export CAP_BASE=$FOLD/figs
    export CAP_REL=U4Mesh_test2
    export CAP_STEM=U4Mesh_test2_${GEOM}
    case $arg in  
       pvcap) source pvcap.sh cap  ;;  
       mpcap) source mpcap.sh cap  ;;  
       pvpub) source pvcap.sh env  ;;  
       mppub) source mpcap.sh env  ;;  
    esac

    if [ "$arg" == "pvpub" -o "$arg" == "mppub" ]; then 
        source epub.sh 
    fi  
fi 


exit 0 


