#!/bin/bash -l 

for s in $(seq 0 9) 
do
     SLA=r${s}@ ./cxr_solid.sh 
done 



