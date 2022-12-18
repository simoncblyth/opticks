#!/bin/bash -l 

export N=128
#export N=16

export FORCE=T   # T/R/N
export N1=1.0
export N2=1.5
export AOI=BREWSTER

export EYE=2,-7,2
export LOOK=0,0,-0.25
export POLSCALE=1

export B=${B:-1} # 1:pol 2:alt_pol


source sboundary_test.sh $*


