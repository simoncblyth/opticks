#!/bin/bash -l 

usage(){ cat << EOU
make_demos.sh
===============

Gets the list of geometies from CSGDemoTest.sh including the 
commented out ones. Then for each GEOMETRY invokes CSGDemoTest.sh

EOU
}

msg="=== $BASH_SOURCE :"
geometrys=$(perl -n -e 'm,geometry=(\S*), && print "$1\n" '  CSGDemoTest.sh)

for geometry in $geometrys ; do 
   GEOMETRY=$geometry ./CSGDemoTest.sh 
   [ $? -ne 0 ] && echo $msg FAIL for geometry $geometry  && exit 1  
done 

exit 0 
