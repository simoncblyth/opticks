#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
geoms=$(perl -n -e 'm,geom=(\S*), && print "$1\n" '  xxv.sh)

export SCANNER=xxv_scan.sh 

for geom in $geoms ; do  
   echo GEOM=$geom ./xxv.sh 
   GEOM=$geom ./xxv.sh 
   [ $? -ne 0 ] && echo $msg FAIL for geom $geom  && exit 1   
done 

exit 0 

