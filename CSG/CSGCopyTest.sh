#!/bin/bash -l 

export CSGMaker=INFO
export CSGCopy=INFO

# DUMP_NPS  3-bits bitfield (node,prim,solid)  7:111 6:110 5:101 4:100 3:011 2:010 1:001 0:000 
#export DUMP_NPS=7
#export DUMP_RIDX=6 

lldb__ CSGCopyTest $* 


