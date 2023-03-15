#!/bin/bash -l 

defarg="run_cf"
arg=${1:-$defarg}

if [ "${arg/run}" != "$arg" ]; then 
   N=0 ./U4SimulateTest.sh run
   N=1 ./U4SimulateTest.sh run
fi 

if [ "${arg/cf}" != "$arg" ]; then 
   C2CUT=30 ./U4SimulateTest.sh cf
fi

