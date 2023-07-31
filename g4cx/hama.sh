#!/bin/bash -l 

arg=${1:-ana}

export MOI=Hama:0:1000 FOCUS=-100,-100,100 ISEL=4,9  

~/opticks/g4cx/gxt.sh $arg


