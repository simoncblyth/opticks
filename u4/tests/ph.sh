#!/bin/bash -l 

defarg="run_ph"
#defarg="ph"
arg=${1:-$defarg}

N=${N:-0} ./U4SimulateTest.sh $arg


