#!/bin/bash -l 

#defarg="run_cf"
defarg="cf"
arg=${1:-$defarg}

if [ "${arg/run}" != "$arg" ]; then 
   N=0 ./U4SimulateTest.sh run
   [ $? -ne 0 ] && echo $BASH_SOURCE N=0 RUN ERROR && exit 1

   N=1 ./U4SimulateTest.sh run
   [ $? -ne 0 ] && echo $BASH_SOURCE N=1 RUN ERROR && exit 2

fi 

if [ "${arg/cf}" != "$arg" ]; then 
   C2CUT=30 ./U4SimulateTest.sh cf
   [ $? -ne 0 ] && echo $BASH_SOURCE cf ANA ERROR && exit 3
fi

exit 0 


