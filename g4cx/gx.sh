#!/bin/bash -l 


export QUIET=1

./gxs.sh $* 

./gxt.sh $*

./gxr.sh $*


