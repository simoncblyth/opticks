#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
geometrys=$(perl -n -e 'm,geometry=(\S*), && print "$1\n" '  ../CSG/CSGDemoTest.sh)

for geometry in $geometrys ; do  
   GEOMETRY=$geometry ./cxr_demo.sh 
   [ $? -ne 0 ] && echo $msg FAIL for geometry $geometry  && exit 1   
done 

exit 0 


