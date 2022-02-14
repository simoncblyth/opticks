#!/bin/bash -l 
usage(){ cat << EOU
CSGOptiXRenderTest.sh using tests/CSGOptiXRenderTest.py
==============================================================

tests/CSGOptiXRenderTest.py

1. loads the frame isects (4x4 quad4 for every pixel) written by the last run of 
   the CSGOptiXRenderTest (which is used by cxr scripts including cxr_geochain.sh)

2. selects a region of the frame isects specified by DYDX envvar and saves to 

   DYDX=1,1  (3x3 pixels in middle) 
   DYDX=2,2  (5x5 pixels in middle) 
   DYDX=3,3  (7x7 pixels in middle)

3. saves the selected isects to paths 
   /tmp/$USER/opticks/CSGOptiX/CSGOptiXRenderTest/dy_1_dx_1.npy 


4. subsequently can use CSG/tests/CSGQueryTest.cc CSG/CSGQueryTest.sh with YX envvar 
   to load the selected pixel intersects and rerun them  


EOU
}

${IPYTHON:-ipython} -i tests/CSGOptiXRenderTest.py

