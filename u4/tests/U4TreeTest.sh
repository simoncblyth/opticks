#!/bin/bash -l 

bin=U4TreeTest 
defarg="run_ana"
arg=${1:-$defarg}


source $OPTICKS_HOME/bin/COMMON.sh


if [ "${arg/info}" != "$arg" ]; then 
    vars="BASH_SOURCE GEOM ${GEOM}_GDMLPath"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done
fi 


if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    export FOLD=/tmp/$USER/opticks/U4TreeTest
    ${IPYTHON:-ipython} --pdb -i $bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi 

exit 0 


