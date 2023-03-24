#!/bin/bash -l 

#defarg="ph"
defarg="run_ph"
#defarg="dbg"

arg=${1:-$defarg}

N=${N:-1} ./U4SimulateTest.sh $arg


