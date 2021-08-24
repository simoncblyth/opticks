#!/bin/bash -l 

pkg=CSGOptiX
bin=CSGOptiXSimulate
export OPTICKS_OUTDIR=/tmp/$USER/opticks/$pkg/$bin
mkdir -p $OPTICKS_OUTDIR

export MOI=${MOI:-Hama}
#export CEGS=15:0:15:1000
export CEGS=15:0:15:1000:17700:0:0:2000

$bin




