#!/bin/bash -l 

DIR=$(dirname $BASH_SOURCE)

#defarg="run_pr"
defarg="pr"
arg=${1:-$defarg}

$DIR/U4SimulateTest.sh $arg


