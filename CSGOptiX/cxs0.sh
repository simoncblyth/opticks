#!/bin/bash -l 

pkg=CSGOptiX
bin=CSGOptiXSimulate
export OPTICKS_OUTDIR=/tmp/$USER/opticks/$pkg/$bin
mkdir -p $OPTICKS_OUTDIR

export MOI=${MOI:-Hama}
export CEGS=5:0:5:1000

$bin




