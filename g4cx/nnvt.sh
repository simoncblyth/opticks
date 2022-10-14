#!/bin/bash -l 

cmd=${1:-ana}

export MOI=NNVT:0:1000 
export FOCUS=-100,-100,100 
export ISEL=3,7,9  

~/opticks/g4cx/gxt.sh $cmd


