#!/bin/bash -l 


pkg=CSGOptiX
bin=CSGOptiXSimulate


export MOI=${MOI:-Hama}
export CEGS=5:0:5:1000

$bin




