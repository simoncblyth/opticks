#!/bin/bash -l 

arg=${1:-ana}

N=1

apid=973
#bpid=570
bpid=903

opt=ast,nrm
#opt=idx,nrm

# ndy/pdy : to offset the labels

export TOPLINE="N=$N APID=$apid AOPT=$opt BPID=$bpid BOPT=$opt BGC=yellow FOCUS=0,0,255 ./U4SimtraceTest.sh"
TOPLINE="$TOPLINE $arg"
eval $TOPLINE


