#!/bin/bash -l 
usage(){ cat << EOU
U4SimtraceTest_two_pmt_cf.sh
==============================

Melange : presenting two photons A,B from two geometries S,T 

Uses SVOL, TVOL to select volumes to present:

* SVOL : N=0 PMT on left 
* TVOL : N=1 PMT on right 



EOU
}

arg=${1:-ana}

N=1

apid=-973
bpid=-903

opt=ast,nrm
#opt=idx,nrm


export SVOL=0,1,13:24
export TVOL=2:11

# ndy/pdy : to offset the labels

export TOPLINE="N=$N APID=$apid AOPT=$opt BPID=$bpid BOPT=$opt BGC=yellow FOCUS=0,0,255 ./U4SimtraceTest.sh"
TOPLINE="$TOPLINE $arg"
eval $TOPLINE


