#!/bin/bash -l 


#for s in $(seq 1 4) ; do ./cxr_solid.sh r${s}p ; done 

for s in $(seq 0 9) ; do ./cxr_solid.sh r${s}@ ; done 

#NAMEPREFIX=cxr_solid source ./cxr_rsync.sh 


