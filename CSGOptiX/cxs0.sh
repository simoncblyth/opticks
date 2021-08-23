#!/bin/bash -l 


pkg=CSGOptiX
bin=CSGOptiXSimulate


export MOI=${MOI:-PMT_20inch}
export CEG=10:0:10:1000

$bin




