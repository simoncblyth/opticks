#!/bin/bash -l 

arg=${1:-ana}

export TOPLINE="N=1 BPID=748 BOPT=ast,nrm BGC=yellow FOCUS=0,0,255 ./U4SimtraceTest.sh"
TOPLINE="$TOPLINE $arg"
eval $TOPLINE


