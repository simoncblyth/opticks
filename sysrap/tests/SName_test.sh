#!/bin/bash -l 
usage(){ cat << EOU
SName_test.sh 
===============

1. Loads meshname.txt of the geometry configured by 
   the GEOM envvar thats set in $HOME/.opticks/GEOM/GEOM.sh 

2. Dumps the solid names of the geometry 

EOU
}

name=SName_test 
bin=/tmp/$name

source $HOME/.opticks/GEOM/GEOM.sh 

vars="BASH_SOURCE name GEOM"
for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 

gcc $name.cc -std=c++11 -lstdc++ -I.. -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 

$bin
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 

exit 0
