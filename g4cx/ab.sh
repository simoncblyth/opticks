#!/bin/bash -l 

A_FOLD=$($OPTICKS_HOME/g4cx/gxs.sh fold)
B_FOLD=$($OPTICKS_HOME/u4/u4s.sh fold)

source $OPTICKS_HOME/bin/AB_FOLD.sh 
export A_FOLD
export B_FOLD

ab_defarg="ana"
ab_arg=${1:-$ab_defarg}

if [ "$ab_arg" == "info" ]; then
   vars="BASH_SOURCE OPTICKS_HOME ab_arg ab_defarg A_FOLD B_FOLD"
   for var in $vars ; do printf "%20s : %s \n" $var ${!var} ; done 
fi 

if [ "$ab_arg" == "ana" ]; then
    ${IPYTHON:-ipython} --pdb -i $OPTICKS_HOME/g4cx/tests/G4CXSimulateTest_ab.py
fi




