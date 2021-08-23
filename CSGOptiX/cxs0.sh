#!/bin/bash -l 


pkg=CSGOptiX
bin=CSGOptiXSimulate


export MOI=${MOI:-Hama}
export CEG=5:0:5:1000

$bin




