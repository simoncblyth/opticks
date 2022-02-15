#!/bin/bash -l 

#DUMP=3 ORI=-150,0,0 DIR=1,0,0 CSGQueryTest O


#DUMP=3 ORI=-150,1,0 DIR=1,0,0 CSGQueryTest O

#DUMP=3 ORI=-150,-1,0 DIR=1,0,0 CSGQueryTest O


# this is giving hit when miss expected
#DUMP=3 ORI=1,1,200 DIR=0,0,-1 CSGQueryTest O

#DUMP=3 ORI=-1,-1,200 DIR=0,0,-1 CSGQueryTest O

#DUMP=3 ORI=-100,-100,100 DIR=1,1,-1 CSGQueryTest O

#DUMP=3 ORI=-100,-100,0 DIR=1,1,0 CSGQueryTest O

#yx=1,1
#YX=${YX:-$yx} CSGQueryTest


#DUMP=3 ORI=-1,1,0 DIR=1,0,0 CSGQueryTest O

#DUMP=3 ORI=100,10,0 DIR=-1,0,0 CSGQueryTest O


export GEOM=OverlapBoxSphere
DUMP=3 ORI=0,0,0 DIR=1,0,0 CSGQueryTest O



