#!/bin/bash -l 

export IDPath=/usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/f9225f882628d01e0303b3609013324e/1

if [ -z "$QUIET" ]; then 
    if [ ! -d "$IDPath" ]; then 
       echo === $BASH_SOURCE : IDPath directory DOES NOT EXIST : $IDPath 
    else
       echo === $BASH_SOURCE : IDPath directory exists : $IDPath 
    fi 
fi

