#!/bin/bash -l

export IDPATH2=/usr/local/opticks-cmake-overhaul/geocache/CX4GDMLTest_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1

dif(){  diff -y $IDPATH/$1 $IDPATH2/$1 ; }

dif GItemList/GMaterialLib.txt 


ipython -i geocache.py 

