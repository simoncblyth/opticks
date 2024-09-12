#!/bin/bash 

name=scpp_test
FOLD=/tmp/$name 
mkdir -p $FOLD

stds="c++11 c++17"

for std in $stds ; do 
   bin=$FOLD/${name}_$std
   gcc $name.cc -std=$std -lstdc++ -o $bin 
   [ $? -ne 0 ] && echo build error && exit 1 
   $bin
   rc=$?
   echo $bin rc $rc
done 




