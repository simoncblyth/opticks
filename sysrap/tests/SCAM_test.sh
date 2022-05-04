#!/bin/bash -l 

msg=" === $FUNCNAME :"
name=SCAM_test 

gcc $name.cc -g -std=c++11 -lstdc++ -I.. -o /tmp/$name 
[ $? -ne 0 ] && echo $msg compile error && exit 1 



CAM=perspective /tmp/$name
[ $? -ne 0 ] && echo $msg run error && exit 2 

CAM=orthographic /tmp/$name
[ $? -ne 0 ] && echo $msg run error && exit 2 

CAM=0 /tmp/$name
[ $? -ne 0 ] && echo $msg run error && exit 2 



exit 0 

