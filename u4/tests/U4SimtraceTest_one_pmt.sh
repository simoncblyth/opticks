#!/bin/bash -l 

arg=${1:-ana}

LAYOUT=one_pmt
N=${N:-1}

export TOPLINE="N=$N LAYOUT=$LAYOUT FOCUS=0,10,185 ./U4SimtraceTest.sh"
TOPLINE="$TOPLINE $arg"
eval $TOPLINE


