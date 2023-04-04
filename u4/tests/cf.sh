#!/bin/bash -l 
usage(){ cat << EOU
cf.sh  : runs U4SimulateTest.sh twice with N=0 and N=1 and Chi2 compares histories
======================================================================================

::

    u4t                              # cd ~/opticks/u4/tests 

    APID=475 BPID=476 ./cf.sh cf     # dump single photons and chi2 compare

    ./cf.sh run_cf                   # create SEvt and compare histories  

    POM=0 ./cf.sh                    # traditional stop at photocathode 

    NUM_PHOTONS=50000 POM=0 ./cf.sh 


EOU
}

defarg="run_cf"
#defarg="cf"
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


