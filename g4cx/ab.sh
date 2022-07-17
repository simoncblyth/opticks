#!/bin/bash -l 

usage(){ cat << EON
ab.sh
=========

::

    ./ab.sh info 
    ./ab.sh ab
    ./ab.sh recplot


EON
}

A_FOLD=$($OPTICKS_HOME/g4cx/gxs.sh fold)
B_FOLD=$($OPTICKS_HOME/u4/u4s.sh fold)

source $OPTICKS_HOME/bin/AB_FOLD.sh 
export A_FOLD
export B_FOLD

ab_defarg="ab"
ab_arg=${1:-$ab_defarg}
stem=${ab_arg}
script=$OPTICKS_HOME/g4cx/tests/$stem.py


if [ "$ab_arg" == "info" ]; then
   vars="BASH_SOURCE OPTICKS_HOME ab_arg ab_defarg A_FOLD B_FOLD script"
   for var in $vars ; do printf "%20s : %s \n" $var ${!var} ; done 
fi 

if [ -f "$script" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
fi



