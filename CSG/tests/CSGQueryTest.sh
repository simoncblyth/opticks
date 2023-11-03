#!/bin/bash -l 

cd $(dirname $BASH_SOURCE)

source $HOME/.opticks/GEOM/GEOM.sh 

name=CSGQueryTest
#defarg="run_ana"
defarg="run"
arg=${1:-$defarg}


if [ "${arg/run}" != "$arg" ]; then 
   $name $* 
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1
fi

if [ "${arg/ana}" != "$arg" ]; then 
   ${IPYTHON:-ipython} --pdb -i tests/$name.py 
   [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 2
fi

exit 0 



