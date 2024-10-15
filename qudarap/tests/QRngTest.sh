#!/bin/bash 
usage(){ cat << EOU
QRngTest.sh
==============

~/o/qudarap/tests/QRngTest.sh 

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

bin=QRngTest
script=$bin.py 

export FOLD=$TMP/$bin   ## needs to match whats in QRngTest.cc
mkdir -p $FOLD/float
mkdir -p $FOLD/double


defarg="info_run_ana"
arg=${1:-$defarg}


vars="FOLD bin script"

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done
fi 


if [ "${arg/run}" != "$arg" ]; then 
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${PYTHON:-python} $script
fi 

if [ "${arg/pdb}" != "$arg" ]; then 
    ${IPYTHON:-ipython} -i --pdb $script
fi 



