#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
export IDPath=/usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/f9225f882628d01e0303b3609013324e/1

if [ ! -d "$IDPath" ]; then 
   echo $msg IDPath directory DOES NOT EXIST : $IDPath 
else
   echo $msg IDPath directory exists : $IDPath 
fi 

