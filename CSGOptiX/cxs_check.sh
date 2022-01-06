#!/bin/bash -l 

vars="COMP GEOM GBASE moi cegs cfbase gridscale ce_offset ce_scale"
for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}"  ; done 

