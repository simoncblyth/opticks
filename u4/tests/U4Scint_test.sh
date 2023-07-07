#!/bin/bash -l 
usage(){ cat << EOU
U4Scint_test.sh
================

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
name=U4Scint_test

defarg="info_build_run_ana"
arg=${1:-$defarg}

source $HOME/.opticks/GEOM/GEOM.sh

export FOLD=/tmp/$name 
bin=$FOLD/$name 
mkdir -p $FOLD 

clhep-
g4-

vars="BASH_SOURCE SDIR name defarg arg FOLD GEOM"

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
   gcc \
       $SDIR/$name.cc \
       -std=c++11 -lstdc++ \
       -I$HOME/opticks/sysrap \
       -I$(clhep-prefix)/include \
       -I$(g4-prefix)/include/Geant4  \
       -I.. \
       -L$(g4-prefix)/lib \
       -L$(clhep-prefix)/lib \
       -lG4global \
       -lG4geometry \
       -lCLHEP \
       -o $bin

   [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
   case $(uname) in 
      Darwin) lldb__ $bin ;;
      Linux)  gdb__ $bin ;;
   esac
   [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi 

if [ "${arg/ana}" != "$arg" ]; then 
   ${IPYTHON:-ipython} --pdb -i $SDIR/$name.py  
   [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi 


exit 0 

