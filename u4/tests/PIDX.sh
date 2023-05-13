#!/bin/bash -l 

DIR=$(dirname $BASH_SOURCE)
export U4Recorder=INFO
export U4Recorder__ClassifyFake_FindPV_r=1


N=0 PIDX=${PIDX:-128} $DIR/U4SimulateTest.sh run



