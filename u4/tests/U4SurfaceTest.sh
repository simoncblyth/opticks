#!/bin/bash -l 

bin=U4SurfaceTest

defarg="run_ana"
arg=${1:-$defarg}


export GEOM=J006 
export FOLD=/tmp/$bin/$GEOM/surface
mkdir -p $FOLD

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi

if [ "${arg/ls}" != "$arg" ]; then
   find $FOLD -type f 
   echo "--"
   find $FOLD -name NPFold_meta.txt
fi

if [ "${arg/ana}" != "$arg" ]; then

   isub=${ISUB:-5}
   subs=($(ls -1 $FOLD | grep -v "\."))
   sub=${subs[$isub]}  

   FOLD=$FOLD/$sub ${IPYTHON:-ipython} --pdb -i $bin.py 
   [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 2
fi

exit 0 



