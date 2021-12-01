#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
geoms=$(perl -n -e 'm,geom=(nnvt_maker_\S*), && print "$1\n" '  xxs.sh)

export SCANNER=xxs_scan.sh

for geom in $geoms ; do  
   echo GEOM=$geom ./xxs.sh 
   GEOM=$geom ./xxs.sh 
   [ $? -ne 0 ] && echo $msg FAIL for geom $geom  && exit 1   
done 

exit 0 

