#!/bin/bash -l 

export N=128
#export N=16

export FORCE=N   # T/R/N
export N1=1.5
export N2=1.0
export AOI=CRITICAL

export EYE=2,-4,0.7
export LOOK=0,0,0.5
export POLSCALE=1


source sboundary_test.sh $*

