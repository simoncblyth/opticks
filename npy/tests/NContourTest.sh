#!/bin/bash -l 

export GEOM=Cone_0 
export NContourTest_Cone_increase_z2=0.25
export NContourTest_Cone_decrease_z1=0.25

NContourTest 

${IPYTHON:-ipython} -i NContourTest.py  


