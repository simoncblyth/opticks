#!/bin/bash -l 
usage(){ cat << EON
pr.sh
======

When just looking at plots can use argumentless invokation
which will pop up two windows for N=0 and N=1::

   ./pr.sh 

When capturing screenshots for presentation need to use N=0 or N=1 
for the auto-book-keeping to work, eg::

    N=0 ./pr.sh 
    N=1 ./pr.sh 

Then whilst each window is showing::

    ./pr.sh mpcap 
    PUB="some_comment" ./pr.sh mppub

EON
}

DIR=$(dirname $BASH_SOURCE)
export TIGHT=1

#defarg="run_pr"
defarg="pr"
arg=${1:-$defarg}

$DIR/U4SimulateTest.sh $arg

