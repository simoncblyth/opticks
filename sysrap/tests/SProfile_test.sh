#!/bin/bash -l 
usage(){ cat << EOU
SProfile_test.sh
===============================

::

   DELAY=0 ./SProfile_test.sh run_ana 
   DELAY=10 ./SProfile_test.sh run_ana 
   DELAY=20 ./SProfile_test.sh run_ana 
   DELAY=100 ./SProfile_test.sh run_ana 

EOU
}

name=SProfile_test 

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

defarg=build_run_ana
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then
   gcc $name.cc -std=c++11 -lstdc++ -I.. -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then
   DELAY=${DELAY:-10} $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi

if [ "${arg/ana}" != "$arg" ]; then
   ${IPYTHON:-ipython} --pdb -i $name.py 
   [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3 
fi

exit 0 


