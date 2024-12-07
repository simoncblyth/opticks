#!/bin/bash
usage(){ cat << EOU
SEventTest.sh
===============

~/o/sysrap/tests/SEventTest.sh

EOU
}
cd $(dirname $(realpath $BASH_SOURCE))

name=SEventTest 
script=$name.py

export FOLD=$TMP/$name
mkdir -p $FOLD

defarg=info_run_ls_ana
arg=${1:-$defarg}

#test=ALL
test=MakeTorchGenstep
export TEST=${TEST:-$test}


if [ "$TEST" == "MakeTorchGenstep" ]; then
   export SEvent__MakeGenstep_num_ph=1000000
   export SEvent__MakeGenstep_num_gs=10
fi 

vars="name script FOLD defarg arg TEST"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/run}" != "$arg" ]; then
   $name
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi

if [ "${arg/ls}" != "$arg" ]; then
   echo ls -alst $FOLD
   ls -alst $FOLD
   [ $? -ne 0 ] && echo $BASH_SOURCE ls error && exit 1 
fi 

if [ "${arg/pdb}" != "$arg" ]; then
   ${IPYTHON:-ipython} --pdb -i $script
   [ $? -ne 0 ] && echo $BASH_SOURCE pdb error && exit 1 
fi
 
if [ "${arg/ana}" != "$arg" ]; then
   ${PYTHON:-python} $script
   [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 1 
fi
 
exit 0 


