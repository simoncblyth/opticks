#!/bin/bash -l 

name=SDir_test 

gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name 
[ $? -ne 0 ] && echo $msg compile error && exit 1 

export IDPath=/usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/f9225f882628d01e0303b3609013324e/1

/tmp/$name
[ $? -ne 0 ] && echo $msg run error && exit 2 

exit 0 

