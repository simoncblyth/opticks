#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
name=OpticksPhotonSTANDALONETest 

gcc $name.cc -I.. -std=c++11 -lstdc++ -o /tmp/$name 
[ $? -ne 0 ] && echo $msg compile error && exit 1 

/tmp/$name
[ $? -ne 0 ] && echo $msg run error && exit 2

exit 0 

