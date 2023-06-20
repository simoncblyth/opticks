#!/bin/bash -l 
usage(){ cat << EOU
QPMT_vs_SPMT_Test.sh
======================

EOU
}
name=QPMT_vs_SPMT_Test

export SCRIPT=$(basename $BASH_SOURCE) 
REALDIR=$(cd $(dirname $BASH_SOURCE) && pwd)

#export QFOLD=/tmp/QPMT_Test
export QFOLD=/tmp/QPMTTest
export SFOLD=/tmp/SPMT_test

defarg="info_ana"
arg=${1:-$defarg}

vars="name SCRIPT REALDIR QFOLD SFOLD"

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $REALDIR/$name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 1 
fi 

exit 0 
