#!/bin/bash -l 

source ../../bin/GEOM_.sh 

export CFBASE

inst=0,1,2,3
export INST=${INST:-$inst}

SCFTest 

