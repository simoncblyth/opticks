#!/bin/bash -l 
usage(){ cat << EOU
CSGNodeScanTest.sh
=====================


EOU
}

defarg="run_ana"
arg=${1:-$defarg}
bin=CSGNodeScanTest

#geom=iphi
geom=acyl
export GEOM=${GEOM:-$geom}
export FOLD=/tmp/$USER/opticks/$bin/$GEOM


if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/ana}" != "$arg" ]; then
   ${IPYTHON:-ipython} --pdb -i $bin.py 
   [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 2
fi 

exit 0 

